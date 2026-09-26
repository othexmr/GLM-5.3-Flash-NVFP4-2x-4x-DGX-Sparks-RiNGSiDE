"""Opt-in TP4 small all-reduce using two direct-neighbour RDMA pair contexts.

The host proxy and pinned-memory layout are adapted from Apache-2.0
b12x.comm.roce. No opposite physical rank is ever connected. Runtime
construction is collective; a failure after the first enqueue is fatal.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from ._lean_cute import PACK_BYTES, get_launcher, is_launcher_prepared
from ._proxy import Layout, Proxy
from .protocol import WORLD, pair_local_rank as _pair_local_rank
from .protocol import pair_map as _pair_map
from .protocol import peer as _peer

LOG = logging.getLogger(__name__)
MAX_BYTES = 2 * 1024 * 1024  # v2: 256 KiB -> 2 MiB to find the crossover (v1 was still 3.2x faster at 256 KiB)
# v4 (2026-09-23): messages of at least GLM53_LEAN_CHUNK_MIN_BYTES run as GLM53_LEAN_CHUNKS pipelined logical ops
# in one launch (the proxy holds 4 slots = 2 x the largest chunk count). Default 1 = the v2 schedule.
MAX_CHUNKS = 2
HCA_NAMES = ("rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1")
_DTYPE_NAME = {
    torch.bfloat16: "bfloat16", torch.float16: "float16",
    torch.float32: "float32",
}


def _gather(group, value):
    items = [None] * WORLD
    dist.all_gather_object(items, value, group=group)
    return items


def _vote(group, error: str | None, stage: str) -> None:
    errors = _gather(group, error)
    bad = [f"rank {r}: {e}" for r, e in enumerate(errors) if e]
    if bad:
        raise RuntimeError(f"lean TP4 {stage} failed: " + "; ".join(bad))


def _grid(size_packs: int, threads: int, max_blocks: int) -> int:
    required = max(1, (size_packs + 2 * threads - 1) // (2 * threads))
    return min(1 << (required - 1).bit_length(), max_blocks)


class LeanTp4AllReduce:
    """One transport kernel per eligible tensor; two physical neighbour hops."""

    def __init__(self, group, device: torch.device | str | int = "cuda:0",
                 max_bytes: int = MAX_BYTES, threads: int = 512,
                 blocks: int = 8, spin_limit: int = 20_000_000,
                 fatal_on_error: bool = False, chunks: int | None = None,
                 chunk_min_bytes: int | None = None) -> None:
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_bytes = int(max_bytes)
        self.threads = int(threads)
        self.blocks = int(blocks)
        self.spin_limit = int(spin_limit)
        self.fatal_on_error = bool(fatal_on_error)
        self.chunks = int(os.environ.get("GLM53_LEAN_CHUNKS", "1") if chunks is None else chunks)
        self.chunk_min_bytes = int(os.environ.get("GLM53_LEAN_CHUNK_MIN_BYTES", "262144")
                                   if chunk_min_bytes is None else chunk_min_bytes)
        self._lock = threading.Lock()
        self._stop_watchdog = threading.Event()
        self._closed = False
        self._proxies: list[Proxy] = []
        self._regions: list[torch.Tensor] = []
        self._control = []
        self._last_stream = None
        self._event = None
        self._capture_id = 0
        self._capture_stream = None
        self._scratch = None

        error = None
        try:
            if self.world_size != WORLD or self.rank not in range(WORLD):
                raise ValueError("TP4 only")
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("CUDA device required")
            if not getattr(torch.cuda.get_device_properties(self.device), "is_integrated", False):
                raise ValueError("integrated GPU with directly mapped pinned memory required")
            if tuple(Path("/sys/class/infiniband").glob("*")) == ():
                raise ValueError("no RDMA device visible")
            if self.max_bytes < PACK_BYTES or self.max_bytes > MAX_BYTES:
                raise ValueError(f"max_bytes must be 16..{MAX_BYTES}")
            if self.threads % 32 or self.threads < 32 or self.threads > 1024:
                raise ValueError("threads must be a warp multiple in 32..1024")
            if self.blocks < 1 or self.blocks & (self.blocks - 1):
                raise ValueError("blocks must be a power of two")
            if self.spin_limit < 1:
                raise ValueError("spin_limit must be positive")
            if self.chunks not in (1, MAX_CHUNKS) or self.chunk_min_bytes < 64 * 1024:
                raise ValueError(f"GLM53_LEAN_CHUNKS must be 1 or {MAX_CHUNKS}; chunk minimum >= 64 KiB")
            gid = int(os.environ.get("NCCL_IB_GID_INDEX", "3"))
            if gid < 0:
                raise ValueError("invalid NCCL_IB_GID_INDEX")
            for name in HCA_NAMES:
                if not (Path("/sys/class/infiniband") / name).exists():
                    raise ValueError(f"missing RDMA device {name}")
        except Exception as exc:
            error = str(exc)
        _vote(group, error, "preflight")

        self.slot_bytes = (self.max_bytes + 4095) // 4096 * 4096
        self._counter_classes = self.blocks.bit_length()
        blobs: list[bytes] = []
        error = None
        try:
            with torch.cuda.device(self.device):
                self._layout = Layout(2, self.slot_bytes)
                self._event = torch.cuda.Event()
                for phase in (1, 2):
                    region = torch.zeros(self._layout.total_bytes, dtype=torch.uint8, pin_memory=True)
                    host_ptr = region.data_ptr()
                    from cuda.bindings import runtime as cudart
                    rc, device_ptr = cudart.cudaHostGetDevicePointer(host_ptr, 0)
                    if rc != cudart.cudaError_t.cudaSuccess or int(device_ptr) != host_ptr:
                        raise RuntimeError("pinned region must have identical host/device address")
                    proxy = Proxy(
                        world_size=2, rank=_pair_local_rank(self.rank, phase),
                        hca_names=HCA_NAMES, gid_index=gid,
                        region_ptr=host_ptr, region_bytes=self._layout.total_bytes,
                        slot_bytes=self.slot_bytes, peer_hca_map=_pair_map(self.rank, phase),
                        opposite_paths=2,
                    )
                    self._regions.append(region)
                    self._proxies.append(proxy)
                    ctrl = region[self._layout.ctrl_off:self._layout.ctrl_off + 32].view(torch.int32)
                    self._control.append(ctrl.numpy())
                    blobs.append(proxy.local_blob())
                if self._layout.slots < 2 * MAX_CHUNKS:
                    raise RuntimeError(f"proxy has {self._layout.slots} slots; v4 needs {2 * MAX_CHUNKS}")
                # [epoch, stage1[class][chunk], stage2[class][chunk], tail[class], poison]
                self._counters = torch.zeros(
                    2 + (2 * MAX_CHUNKS + 1) * self._counter_classes, dtype=torch.int32, device=self.device)
        except Exception as exc:
            error = repr(exc)
        try:
            _vote(group, error, "allocation")
        except Exception:
            self.close()
            raise

        statuses = _gather(group, (tuple(blobs), self.slot_bytes, self.max_bytes,
                                   self.spin_limit, self.threads, self.blocks, self.chunks, self.chunk_min_bytes))
        errors = []
        for r, (_, *geometry) in enumerate(statuses):
            if tuple(geometry) != (self.slot_bytes, self.max_bytes, self.spin_limit, self.threads, self.blocks,
                                   self.chunks, self.chunk_min_bytes):
                errors.append(f"rank {r} geometry differs")
        if errors:
            self.close()
            raise RuntimeError("; ".join(errors))
        error = None
        try:
            for phase, proxy in enumerate(self._proxies, 1):
                peer = _peer(self.rank, phase)
                local = _pair_local_rank(self.rank, phase)
                pair_blobs = [None, None]
                pair_blobs[local] = blobs[phase - 1]
                pair_blobs[1 - local] = statuses[peer][0][phase - 1]
                proxy.connect(pair_blobs)
            for proxy in self._proxies:
                proxy.start()
        except Exception as exc:
            error = repr(exc)
        try:
            _vote(group, error, "connect")
        except Exception:
            self.close()
            raise
        LOG.info("lean TP4 ready rank=%s peers=(%s,%s) max_bytes=%s chunks=%s chunk_min_bytes=%s slots=%s",
                 self.rank, _peer(self.rank, 1), _peer(self.rank, 2), self.max_bytes, self.chunks,
                 self.chunk_min_bytes, self._layout.slots)
        if self.fatal_on_error:
            self._watchdog = threading.Thread(
                target=self._watch_health, name=f"lean-tp4-watch-rank{self.rank}", daemon=True)
            self._watchdog.start()

    def _watch_health(self) -> None:
        # Captured graph replays do not call Python all_reduce. A poisoned
        # graph must kill the worker rather than return an unwritten output.
        while not self._stop_watchdog.wait(0.01):
            try:
                self.check_health()
            except Exception as exc:
                LOG.critical("lean TP4 transport failed: %s; terminating worker", exc)
                os._exit(86)

    def _bases(self, phase: int) -> tuple[int, int, int, int]:
        ptr = self._regions[phase - 1].data_ptr()
        layout = self._layout
        return ptr + layout.recv_off, ptr + layout.flag_off, ptr + layout.send_off, ptr + layout.ctrl_off

    def _key(self, dtype: torch.dtype, chunks: int = 1) -> tuple[object, ...]:
        return (_DTYPE_NAME[dtype], self.rank, self.threads,
                self._layout.slots, self._layout.flag_stride, chunks, self.device.index)

    def chunks_for(self, nbytes: int) -> int:
        return self.chunks if self.chunks > 1 and nbytes >= self.chunk_min_bytes else 1

    def prepare(self, dtypes: Sequence[torch.dtype] = (torch.bfloat16,)) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare before graph capture")
        with torch.cuda.device(self.device):
            for dtype in dtypes:
                for chunks in sorted({1, self.chunks}):
                    get_launcher(*self._key(dtype, chunks))
            self._ensure_scratch()

    def _ensure_scratch(self) -> None:
        if self._scratch is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("alignment scratch must be prepared before capture")
            self._scratch = (
                torch.empty(self.max_bytes, dtype=torch.uint8, device=self.device),
                torch.empty(self.max_bytes, dtype=torch.uint8, device=self.device),
            )

    def should_allreduce(self, tensor: torch.Tensor) -> bool:
        if self._closed:
            raise RuntimeError("lean TP4 runtime closed")
        nbytes = tensor.numel() * tensor.element_size()
        return (tensor.is_cuda and tensor.device == self.device and tensor.is_contiguous()
                and tensor.dtype in _DTYPE_NAME and 0 < nbytes <= self.max_bytes
                and nbytes % PACK_BYTES == 0)

    def check_health(self) -> None:
        for phase, proxy in enumerate(self._proxies, 1):
            if proxy.failed():
                raise RuntimeError(f"lean TP4 phase {phase} proxy failed: {proxy.error()}")
            if self._control and int(self._control[phase - 1][2]):
                seq = int(self._control[phase - 1][2])
                raise RuntimeError(f"lean TP4 phase {phase} timeout at sequence {seq}; runtime poisoned")

    def _order(self, capturing: bool) -> None:
        current = torch.cuda.current_stream(self.device)
        if capturing:
            from cuda.bindings import runtime as cudart
            info = cudart.cudaStreamGetCaptureInfo(current.cuda_stream)
            if info[0] != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("cannot identify CUDA capture")
            capture_id = int(info[2])
            if capture_id != self._capture_id:
                self._capture_id, self._capture_stream = capture_id, current
            elif current != self._capture_stream:
                raise RuntimeError("all lean TP4 collectives in one capture must share a stream")
        elif self._last_stream is not None and current != self._last_stream:
            current.wait_event(self._event)

    def all_reduce(self, tensor: torch.Tensor, chunks: int | None = None) -> torch.Tensor:
        """`chunks` forces 1 or the configured count for one call (the in-process A/B leaf); None applies the policy."""
        with self._lock:
            self.check_health()
            if not self.should_allreduce(tensor):
                raise ValueError("ineligible lean TP4 tensor")
            with torch.cuda.device(self.device):
                capturing = torch.cuda.is_current_stream_capturing()
                nbytes = tensor.numel() * tensor.element_size()
                if chunks is None:
                    chunks = self.chunks_for(nbytes)
                elif chunks not in (1, self.chunks) or (chunks > 1 and nbytes < 64 * 1024):
                    raise ValueError("forced chunks must be 1 or the configured count, and >= 64 KiB when chunked")
                key = self._key(tensor.dtype, chunks)
                if capturing and not is_launcher_prepared(*key):
                    raise RuntimeError("lean TP4 launcher not prepared before capture")
                launch = get_launcher(*key)
                self._order(capturing)
                out = torch.empty_like(tensor)
                src, dst = tensor, out
                if tensor.data_ptr() % PACK_BYTES:
                    self._ensure_scratch()
                    src = self._scratch[0][:nbytes].view(tensor.dtype).view(tensor.shape)
                    src.copy_(tensor)
                if out.data_ptr() % PACK_BYTES:
                    self._ensure_scratch()
                    dst = self._scratch[1][:nbytes].view(tensor.dtype).view(tensor.shape)
                grid = _grid(nbytes // PACK_BYTES, self.threads, self.blocks)
                klass = grid.bit_length() - 1
                base = self._counters.data_ptr()
                classes = self._counter_classes
                # Per (grid class, chunk) staging counters: a launch adds exactly `grid` to each chunk it uses.
                stage1 = base + 4 * (1 + klass * MAX_CHUNKS)
                stage2 = base + 4 * (1 + classes * MAX_CHUNKS + klass * MAX_CHUNKS)
                tail = base + 4 * (1 + 2 * classes * MAX_CHUNKS + klass)
                poison = base + 4 * (1 + (2 * MAX_CHUNKS + 1) * classes)
                launch(src.data_ptr(), dst.data_ptr(), nbytes // PACK_BYTES,
                       nbytes, *self._bases(1), *self._bases(2), self.slot_bytes,
                       base, stage1, stage2, tail, poison, self.spin_limit, grid)
                if dst is not out:
                    out.copy_(dst)
                if not capturing:
                    current = torch.cuda.current_stream(self.device)
                    self._event.record(current)
                    self._last_stream = current
            if not capturing:
                self.check_health()
            return out

    def stats(self) -> dict:
        return {"rank": self.rank, "epoch": int(self._counters[0].item()), "chunks": self.chunks,
                "chunk_min_bytes": self.chunk_min_bytes, "slots": self._layout.slots,
                "phase1": self._proxies[0].stats(), "phase2": self._proxies[1].stats(),
                "phase1_paths": self._proxies[0].path_counters(),
                "phase2_paths": self._proxies[1].path_counters()}

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop_watchdog.set()
            watchdog = getattr(self, "_watchdog", None)
            if watchdog is not None and watchdog is not threading.current_thread():
                watchdog.join()
            with contextlib.suppress(Exception):
                torch.cuda.synchronize(self.device)
            for proxy in self._proxies:
                proxy.close()
            self._proxies.clear()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
