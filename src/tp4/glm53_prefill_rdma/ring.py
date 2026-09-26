# SPDX-License-Identifier: Apache-2.0
"""Two-direction prefill ring collectives over our RDMA transport (rdma-prefill lane, 2026-09-24).

`PrefillRing` owns one rank's registered host region, four RC queue pairs (a send and a receive QP per direction, one
PCIe function per direction), a proxy thread and the CUDA kernel of `_prb.cu` (prebuilt as `libprb-<source sha16>.so`
next to this file, or at GLM53_PRB_SO). Construction is collective over the TP CPU group and every stage is voted, so
either all four ranks have the runtime or none does. After construction a failure never falls back: the watchdog
turns a failed proxy or a timed-out GPU wait into a process exit (code 86), like the lean all-reduce.

    ring = PrefillRing(cpu_group, rank, device)
    ring.reduce_scatter(out[P/4, 4096], inp[P, 4096])   # NCCL's association, bit-identical result
    ring.all_gather(out[P, 4096], inp[P/4, 4096])

Same class name and calls as the one-way runtime (outputs/2026-09-23-rdma-prefill-collectives), so the ownership
helper's GLM53_MHC_PREFILL_RDMA hook drives either package unchanged.
"""
from __future__ import annotations

import contextlib
import ctypes
import hashlib
import logging
import os
import threading
from pathlib import Path

from .protocol import HCA_NAMES, HIDDEN, KIND_AG, KIND_RS, MEM_HOSTTHP, MEM_PINNED, WORLD

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "_prb.cu"
ABI = 2
VERSION = "bidir-1"
MEMORY = {"pinned": MEM_PINNED, "hostthp": MEM_HOSTTHP}
# defaults (chunk rows, slots per direction, parts per chunk, CTAs, memory): the configuration the exploration sweep
# picked by its pre-declared rule (probe-1, 2026-09-24 03:37: "grid32"); see the README
DEFAULTS = dict(chunk_rows=32, slots=16, parts=4, grid=32, memory="hostthp")


def source_digest() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:16]


def library_path() -> Path:
    override = os.environ.get("GLM53_PRB_SO")
    return Path(override) if override else HERE / f"libprb-{source_digest()}.so"


_LIB = None
_LIB_LOCK = threading.Lock()


def load() -> ctypes.CDLL:
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        path = library_path()
        if not path.exists():
            raise RuntimeError(f"prefill ring library {path} is missing")
        lib = ctypes.CDLL(str(path))
        u64, u32, p, i = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int
        lib.prb_abi.restype = i
        lib.prb_blob_bytes.restype = u64
        lib.prb_layout_query.argtypes = [u64, u64, u64, ctypes.POINTER(u64)]
        lib.prb_layout_query.restype = i
        lib.prb_geometry.argtypes = [i, u32, u32, ctypes.POINTER(u32)]
        lib.prb_geometry.restype = i
        lib.prb_item_at.argtypes = [i, u32, u32, u32, ctypes.POINTER(u32)]
        lib.prb_item_at.restype = i
        lib.prb_send_step.argtypes = [i, u32, u32, i, u32, ctypes.POINTER(ctypes.c_int32)]
        lib.prb_send_step.restype = i
        lib.prb_create.argtypes = [i, u64, u64, u64, u32, i, i, i, u64, i, ctypes.c_char_p, u64]
        lib.prb_create.restype = p
        lib.prb_local_blob.argtypes = [p, p, u64]
        lib.prb_local_blob.restype = i
        lib.prb_connect.argtypes = [p, p, p]
        lib.prb_connect.restype = i
        lib.prb_start.argtypes = [p]
        lib.prb_start.restype = i
        lib.prb_launch.argtypes = [p, i, p, p, u64, p]
        lib.prb_launch.restype = i
        lib.prb_health.argtypes = [p, ctypes.c_char_p, u64]
        lib.prb_health.restype = i
        lib.prb_error.argtypes = [p]
        lib.prb_error.restype = ctypes.c_char_p
        lib.prb_region.argtypes = [p]
        lib.prb_region.restype = p
        lib.prb_stats.argtypes = [p, ctypes.POINTER(u64)]
        lib.prb_stats.restype = None
        lib.prb_stop.argtypes = [p]
        lib.prb_stop.restype = None
        lib.prb_destroy.argtypes = [p]
        lib.prb_destroy.restype = None
        if lib.prb_abi() != ABI:
            raise RuntimeError(f"prefill ring ABI {lib.prb_abi()} != {ABI}")
        _LIB = lib
        return lib


def region_backing(address: int) -> dict:
    """Size and AnonHugePages of the mapping that holds `address` (/proc/self/smaps), for the READY log line."""
    try:
        cur = None
        for line in Path("/proc/self/smaps").read_text().splitlines():
            head = line.split()
            if head and "-" in head[0] and len(head) >= 5 and all(c in "0123456789abcdef-" for c in head[0]):
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                cur = {"lo": lo, "hi": hi} if lo <= address < hi else None
                continue
            if cur is not None and head and head[0] in ("Size:", "AnonHugePages:", "Rss:"):
                cur[head[0][:-1]] = int(head[1])
                if head[0] == "AnonHugePages:":
                    return {"size_kib": cur.get("Size"), "rss_kib": cur.get("Rss"), "anon_huge_kib": cur["AnonHugePages"]}
    except OSError:
        pass
    return {}


def _gather(group, value):
    import torch.distributed as dist
    items = [None] * WORLD
    dist.all_gather_object(items, value, group=group)
    return items


def _vote(group, error, stage):
    errors = _gather(group, error)
    bad = [f"rank {r}: {e}" for r, e in enumerate(errors) if e]
    if bad:
        raise RuntimeError(f"prefill ring {stage} failed: " + "; ".join(bad))


class PrefillRing:
    def __init__(self, cpu_group, rank: int, device, *, max_rows: int = 14336, chunk_rows: int | None = None,
                 nslot: int | None = None, parts: int | None = None, grid: int | None = None,
                 proxy_cpu: int | None = None, timeout_s: float | None = None, memory: str | None = None,
                 fatal_on_error: bool = True):
        import torch
        import torch.distributed as dist
        env = os.environ
        self.group = cpu_group
        self.rank = int(rank)
        self.device = torch.device(device)
        self.max_rows = int(max_rows)
        self.chunk_rows = int(env.get("GLM53_PRB_CHUNK_ROWS", DEFAULTS["chunk_rows"]) if chunk_rows is None else chunk_rows)
        self.nslot = int(env.get("GLM53_PRB_SLOTS", DEFAULTS["slots"]) if nslot is None else nslot)
        self.parts = int(env.get("GLM53_PRB_PARTS", DEFAULTS["parts"]) if parts is None else parts)
        self.grid = int(env.get("GLM53_PRB_GRID", DEFAULTS["grid"]) if grid is None else grid)
        self.proxy_cpu = int(env.get("GLM53_PRB_PROXY_CPU", "-1") if proxy_cpu is None else proxy_cpu)
        self.timeout_s = float(env.get("GLM53_PRB_TIMEOUT_S", "300") if timeout_s is None else timeout_s)
        self.memory = str(env.get("GLM53_PRB_MEMORY", DEFAULTS["memory"]) if memory is None else memory)
        self.fatal_on_error = bool(fatal_on_error)
        self._ctx = None
        self._lib = None
        self._lock = threading.Lock()
        self._closed = False
        self._stop = threading.Event()
        self._watchdog = None
        self._event = None
        self._last_stream = None
        self.calls = 0
        self.backing = {}

        error = None
        try:
            if dist.get_world_size(cpu_group) != WORLD or dist.get_rank(cpu_group) != self.rank:
                raise ValueError("prefill ring is TP4 only (group rank must equal the TP rank)")
            if self.device.type != "cuda" or self.device.index is None:
                raise ValueError("an indexed CUDA device is required")
            if not getattr(torch.cuda.get_device_properties(self.device), "is_integrated", False):
                raise ValueError("integrated GPU (host memory at device speed) required")
            for name in HCA_NAMES:
                if not (Path("/sys/class/infiniband") / name).exists():
                    raise ValueError(f"missing RDMA device {name}")
            if (self.max_rows % WORLD or not 1 <= self.chunk_rows <= 1024 or self.nslot < 2 or not 1 <= self.parts <= 64
                    or not 1 <= self.grid <= 64 or self.memory not in MEMORY):
                raise ValueError("invalid prefill ring geometry")
            self._lib = load()
        except Exception as exc:
            error = repr(exc)
        _vote(cpu_group, error, "preflight")

        error = None
        blob = None
        try:
            gid = int(env.get("NCCL_IB_GID_INDEX", "3"))
            err = ctypes.create_string_buffer(512)
            with torch.cuda.device(self.device):
                self._ctx = self._lib.prb_create(self.rank, self.max_rows, self.chunk_rows, self.nslot, self.parts, gid,
                                                 self.grid, self.proxy_cpu, int(self.timeout_s * 1e9),
                                                 MEMORY[self.memory], err, len(err))
            if not self._ctx:
                raise RuntimeError(err.value.decode(errors="replace"))
            n = int(self._lib.prb_blob_bytes())
            buf = ctypes.create_string_buffer(n)
            if self._lib.prb_local_blob(self._ctx, buf, n):
                raise RuntimeError("prb_local_blob failed")
            blob = buf.raw
            self.backing = region_backing(int(self._lib.prb_region(self._ctx)))
        except Exception as exc:
            error = repr(exc)
        try:
            _vote(cpu_group, error, "create")
        except Exception:
            self.close()
            raise

        blobs = _gather(cpu_group, (blob, self.chunk_rows, self.nslot, self.parts, self.grid, self.max_rows,
                                    self.timeout_s, self.memory, VERSION))
        error = None
        try:
            geometry = {tuple(b[1:]) for b in blobs}
            if len(geometry) != 1:
                raise RuntimeError(f"ranks disagree on the ring geometry: {sorted(geometry)}")
            nxt, prv = blobs[(self.rank + 1) % WORLD][0], blobs[(self.rank + WORLD - 1) % WORLD][0]
            if self._lib.prb_connect(self._ctx, nxt, prv):
                raise RuntimeError(self._lib.prb_error(self._ctx).decode(errors="replace"))
            if self._lib.prb_start(self._ctx):
                raise RuntimeError(self._lib.prb_error(self._ctx).decode(errors="replace"))
        except Exception as exc:
            error = repr(exc)
        try:
            _vote(cpu_group, error, "connect")
        except Exception:
            self.close()
            raise
        self._event = torch.cuda.Event()
        if self.fatal_on_error:
            self._watchdog = threading.Thread(target=self._watch, name=f"prefill-ring-watch-{self.rank}", daemon=True)
            self._watchdog.start()
        LOG.warning("GLM53_PRB_READY version=%s rank=%d next=%d prev=%d max_rows=%d chunk_rows=%d slots=%d parts=%d "
                    "grid=%d proxy_cpu=%d memory=%s backing=%s library=%s", VERSION, self.rank,
                    (self.rank + 1) % WORLD, (self.rank + WORLD - 1) % WORLD, self.max_rows, self.chunk_rows,
                    self.nslot, self.parts, self.grid, self.proxy_cpu, self.memory, self.backing, library_path().name)

    # ---------------------------------------------------------------------------------------------- health
    def health(self) -> tuple[int, str]:
        if self._ctx is None:
            return 1, "closed"
        msg = ctypes.create_string_buffer(640)
        code = int(self._lib.prb_health(self._ctx, msg, len(msg)))
        return code, msg.value.decode(errors="replace")

    def check_health(self) -> None:
        code, msg = self.health()
        if code:
            raise RuntimeError(f"prefill ring failed: {msg}")

    def _watch(self) -> None:
        while not self._stop.wait(0.01):
            code, msg = self.health()
            if code:
                LOG.critical("prefill ring transport failed: %s; terminating worker", msg)
                os._exit(86)

    # ---------------------------------------------------------------------------------------------- collectives
    def _launch(self, kind: int, inp, out, rows: int) -> None:
        import torch
        for t in (inp, out):
            if (not t.is_cuda or t.device != self.device or t.dtype != torch.bfloat16 or not t.is_contiguous()
                    or t.dim() != 2 or t.shape[1] != HIDDEN):
                raise RuntimeError("prefill ring needs contiguous [rows, 4096] BF16 tensors on the ring device")
        if rows % WORLD or not 0 < rows <= self.max_rows:
            raise RuntimeError(f"prefill ring rows {rows} unsupported")
        with self._lock:
            if self._closed:
                raise RuntimeError("prefill ring closed")
            current = torch.cuda.current_stream(self.device)
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prefill ring collectives are eager-only")
            if self._last_stream is not None and current != self._last_stream:
                current.wait_event(self._event)
            rc = int(self._lib.prb_launch(self._ctx, kind, inp.data_ptr(), out.data_ptr(), rows, current.cuda_stream))
            if rc:
                code, msg = self.health()
                raise RuntimeError(f"prefill ring launch failed ({rc}): "
                                   f"{self._lib.prb_error(self._ctx).decode(errors='replace')} {msg}")
            self._event.record(current)
            self._last_stream = current
            self.calls += 1

    def reduce_scatter(self, output, inp) -> None:
        """output [P/4, 4096] <- block `rank` of the sum over ranks of inp [P, 4096], in NCCL's association."""
        rows = inp.shape[0]
        if output.shape[0] * WORLD != rows:
            raise RuntimeError("prefill ring reduce-scatter geometry")
        self._launch(KIND_RS, inp, output, rows)

    def all_gather(self, output, inp) -> None:
        """output [P, 4096] <- every rank's inp [P/4, 4096], block r at rows [r*P/4, (r+1)*P/4)."""
        rows = output.shape[0]
        if inp.shape[0] * WORLD != rows:
            raise RuntimeError("prefill ring all-gather geometry")
        self._launch(KIND_AG, inp, output, rows)

    def stats(self) -> dict:
        out = (ctypes.c_uint64 * 16)()
        self._lib.prb_stats(self._ctx, out)
        names = ("seq", "proxy_calls", "posted_fwd", "posted_bwd", "done_fwd", "done_bwd", "bytes_fwd", "bytes_bwd",
                 "send_done_fwd", "send_done_bwd", "stage_next_fwd", "stage_next_bwd", "work_next", "idle_naps",
                 "relays", "memory_kind")
        return dict(zip(names, (int(v) for v in out)))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stop.set()
            if self._watchdog is not None and self._watchdog is not threading.current_thread():
                self._watchdog.join(timeout=1)
        if self._ctx is not None and not self._closed:
            self._closed = True
            with contextlib.suppress(Exception):
                import torch
                torch.cuda.synchronize(self.device)
            self._lib.prb_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
