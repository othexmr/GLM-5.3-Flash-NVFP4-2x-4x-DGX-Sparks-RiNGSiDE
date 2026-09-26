# SPDX-License-Identifier: Apache-2.0
"""Pure-Python contract of the two-direction prefill ring collectives (rdma-prefill lane, 2026-09-24).

Everything `_prb.cu` computes about *where* bytes go, *when* (in which order) and *how* partial sums are associated is
restated here, so the CPU tests can check it without hardware.

Topology (one PCIe function, "PF", per direction; the ceiling's `ring_bi_split`, 25.5-25.6 GB/s per node in hostthp
memory against 24.5 GB/s for the one-way ring):
- forward traffic (rank r -> r+1) runs on lane 0 = the `rocep1s0f*` function (PCIe link A) of the port facing r+1;
- backward traffic (rank r -> r-1) runs on lane 1 = the `roceP2p1s0f*` function (PCIe link B) of the port facing r-1.
Every cable joins the same function index at both ends (measured 2026-09-23 02:10).

Blocks and chunks. A call moves [P, 4096] BF16 rows, P = 4q; block b = rows [b*q, (b+1)*q). A call cuts every block
into `nc` chunks, chunk k = rows [k*q // nc, (k+1)*q // nc) (sizes differ by at most one row, at most `cr_max`), with
nc the smallest multiple of four that keeps chunks within `cr_max` (or q itself for tiny blocks): a multiple of four
balances the two directions exactly.

Reduce-scatter (RS), bit-identical to NCCL's ring. NCCL sums block j as ((x[j+1] + x[j+2]) + x[j+3]) + x[j], rounding
to BF16 after every add. The first `na` = nc/4 chunks of every block (the "A" chunks) take NCCL's forward chain:
  FA0: r sends x_r[r-1] to r+1;  FA1: r sends recv + x_r[r-2];  FA2: r sends recv + x_r[r+1];  out = recv + x_r[r].
The other `nb` = nc - na chunks (the "B" chunks) take a second chain with the same association:
  BB1: r sends x_r[r-2] backward to r-1;  BB2: r sends recv(BB1) + x_r[r-1] backward (= x[j+1] + x[j+2] for j = r-1);
  FB:  r sends x_r[r+1] forward to r+1 (= x[j+3] for j = r+1);  out = (recv(BB2) + recv(FB)) + x_r[r].
Per call every rank sends 3*na + nb = 1.5 nc chunks forward and 2*nb = 1.5 nc backward: the same 3 blocks as the
one-way ring, split evenly over the two PFs.

All-gather (AG), exact copies. r sends its own block to both neighbours (forward in chunk order, backward second half
first), then relays the first nh = nc/2 chunks of block r-1 forward and the other chunks of block r+1 backward, straight
from its receive buffer (no GPU copy). Block r+2 arrives half from each side. 1.5 nc chunks per direction.

Steps. Each direction sends one chunk per "step" (a numbered slot of that direction's send sequence); the receiver
keeps an always-open receive area per (call parity, direction) with one `cr`-row slot and one 4-byte flag per step.
Every dependent send (a reduction or a relay) consumes a chunk its neighbour sent at least one step earlier, and the
GPU work list is the merge of all item streams by step (`streams()`), so every wait points to an earlier item on
every rank: no deadlock by construction (the simulation in test_cpu_protocol.py checks it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

WORLD = 4
HIDDEN = 4096
ROW_BYTES = HIDDEN * 2          # BF16
ROW_PACKS = ROW_BYTES // 16     # 16-byte packs per row
DIRS = 2
FWD, BWD = 0, 1
FLAG_STRIDE = 64                # bytes between flags / ready words (one cache line each)
HCA_NAMES = ("rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1")
KIND_RS, KIND_AG = 1, 2
MEM_PINNED, MEM_HOSTTHP = 0, 1

# source of a send step
SRC_RAW, SRC_RED, SRC_RELAY = 0, 1, 2
# GPU item streams (merged by step): kinds of work
ITEM_STAGE_F, ITEM_STAGE_B, ITEM_OUT_A, ITEM_OUT_B, ITEM_OUT_PREV, ITEM_OUT_NEXT, ITEM_OUT_FARF, ITEM_OUT_FARB = range(8)


def direct_path(a: int, b: int, lane: int) -> int:
    """Local HCA index of `lane` on the cable between neighbouring ranks a and b (the same index at both ends)."""
    lo, hi = min(a, b), max(a, b)
    f1 = (lo, hi) in ((0, 1), (2, 3))
    f0 = (lo, hi) in ((1, 2), (0, 3))
    if not (f0 or f1) or lane not in (0, 1):
        raise ValueError(f"no direct cable between {a} and {b}")
    return (1 if f1 else 0) + 2 * lane


def nxt(rank: int) -> int:
    return (rank + 1) % WORLD


def prv(rank: int) -> int:
    return (rank + WORLD - 1) % WORLD


def send_hca(rank: int, d: int) -> int:
    """HCA that carries rank's outgoing direction-d traffic: lane 0 toward r+1, lane 1 toward r-1."""
    return direct_path(rank, nxt(rank), 0) if d == FWD else direct_path(rank, prv(rank), 1)


def recv_hca(rank: int, d: int) -> int:
    """HCA on which rank receives direction-d traffic (forward from r-1 on lane 0, backward from r+1 on lane 1)."""
    return direct_path(rank, prv(rank), 0) if d == FWD else direct_path(rank, nxt(rank), 1)


# ------------------------------------------------------------------------------------------------ geometry
@dataclass(frozen=True)
class Geometry:
    kind: int
    rows: int     # P, a multiple of 4
    q: int        # rows per block
    cr: int       # receive-slot stride in rows for this call: the largest chunk, ceil(q / nc) <= cr_max
    nc: int       # chunks per block
    na: int       # RS: chunks on NCCL's forward chain
    nb1: int      # RS: B chunks sent raw forward before FA1 (the rest after it)
    nh: int       # AG: chunks of block r-1 relayed forward (the rest of block r+1 goes backward)

    @property
    def nb(self) -> int:
        return self.nc - self.na

    def chunk_rows(self, k: int) -> tuple[int, int]:
        """Rows [c0, c1) of chunk k inside a block (even split, sizes differ by at most one row)."""
        if not 0 <= k < self.nc:
            raise IndexError(k)
        return k * self.q // self.nc, (k + 1) * self.q // self.nc

    def steps(self, d: int) -> int:
        """Send steps of direction d per call."""
        if self.kind == KIND_RS:
            return 3 * self.na + self.nb if d == FWD else 2 * self.nb
        return self.nc + (self.nh if d == FWD else self.nc - self.nh)

    def staged(self, d: int) -> int:
        """Leading steps of direction d whose data the GPU stages (the rest are relays)."""
        return self.steps(d) if self.kind == KIND_RS else self.nc

    def items(self) -> int:
        return 4 * self.nc if self.kind == KIND_RS else 5 * self.nc


def geometry(kind: int, rows: int, cr_max: int) -> Geometry:
    if kind not in (KIND_RS, KIND_AG) or rows <= 0 or rows % WORLD or cr_max < 1:
        raise ValueError("unsupported prefill ring call")
    q = rows // WORLD
    nc = min(4 * math.ceil(math.ceil(q / cr_max) / 4), q)   # chunks per block: a multiple of four unless q is tiny
    cr = math.ceil(q / nc)
    na = nc // 4
    nb1 = (nc - na) // 2
    nh = nc // 2
    return Geometry(kind, rows, q, cr, nc, na, nb1, nh)


@dataclass(frozen=True)
class Step:
    piece: str
    chunk: int              # chunk index inside the block the step carries
    src: int                # SRC_RAW / SRC_RED / SRC_RELAY
    blk_off: int            # block of the local input, relative to rank (RAW/RED)
    dep: int                # neighbour step in the SAME direction this step consumes (RED/RELAY), else -1


def send_step(g: Geometry, d: int, s: int) -> Step:
    """What a rank sends at step s of direction d (identical structure on every rank)."""
    if not 0 <= s < g.steps(d):
        raise IndexError(s)
    na, nb, nb1, nc, nh = g.na, g.nb, g.nb1, g.nc, g.nh
    if g.kind == KIND_RS:
        if d == FWD:
            if s < na:
                return Step("FA0", s, SRC_RAW, -1, -1)
            s -= na
            if s < nb1:
                return Step("FB1", na + s, SRC_RAW, +1, -1)
            s -= nb1
            if s < na:
                return Step("FA1", s, SRC_RED, -2, s)                    # consumes FA0' chunk s (step s)
            s -= na
            if s < nb - nb1:
                return Step("FB2", na + nb1 + s, SRC_RAW, +1, -1)
            s -= nb - nb1
            return Step("FA2", s, SRC_RED, +1, na + nb1 + s)             # consumes FA1' chunk s
        if s < nb:
            return Step("BB1", na + s, SRC_RAW, -2, -1)
        s -= nb
        return Step("BB2", na + s, SRC_RED, -1, s)                       # consumes BB1' chunk (step s)
    if d == FWD:
        if s < nc:
            return Step("G0F", s, SRC_RAW, 0, -1)
        s -= nc
        return Step("G1F", s, SRC_RELAY, 0, s)                           # relays G0F' chunk s (block r-1)
    if s < nc - nh:
        return Step("G0Bhi", nh + s, SRC_RAW, 0, -1)
    s -= nc - nh
    if s < nh:
        return Step("G0Blo", s, SRC_RAW, 0, -1)
    s -= nh
    return Step("G1B", nh + s, SRC_RELAY, 0, s)                          # relays G0Bhi' chunk nh+s (block r+1)


def fb_step(g: Geometry, c: int) -> int:
    """Forward step that carries B chunk c (block chunk na + c) of the FB pieces."""
    return g.na + c if c < g.nb1 else 2 * g.na + c


def streams(g: Geometry) -> list[tuple[int, int, int]]:
    """GPU item streams as (item kind, first step, count); the kernel merges them by step (ties in list order).
    An output item sits one step after the last send it consumes."""
    if g.kind == KIND_RS:
        return [(ITEM_STAGE_F, 0, g.steps(FWD)), (ITEM_STAGE_B, 0, g.steps(BWD)),
                (ITEM_OUT_B, g.nb + 1, g.nb), (ITEM_OUT_A, 2 * g.na + g.nb + 1, g.na)]
    return [(ITEM_STAGE_F, 0, g.nc), (ITEM_STAGE_B, 0, g.nc), (ITEM_OUT_PREV, 1, g.nc), (ITEM_OUT_NEXT, 1, g.nc),
            (ITEM_OUT_FARF, g.nc + 1, g.nh), (ITEM_OUT_FARB, g.nc + 1, g.nc - g.nh)]


def items_before(sts, t: int) -> int:
    return sum(min(max(t - start, 0), count) for _, start, count in sts)


def item_at(g: Geometry, i: int) -> tuple[int, int, int]:
    """Item i of the merged work list -> (item kind, index within its stream, step). Same arithmetic as the kernel."""
    sts = streams(g)
    total = sum(c for _, _, c in sts)
    if not 0 <= i < total:
        raise IndexError(i)
    lo, hi = 0, max(start + count for _, start, count in sts)     # items_before(lo) <= i < items_before(hi)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if items_before(sts, mid) <= i:
            lo = mid
        else:
            hi = mid
    k = i - items_before(sts, lo)
    for kind, start, count in sts:
        if start <= lo < start + count:
            if k == 0:
                return kind, lo - start, lo
            k -= 1
    raise AssertionError("merge arithmetic")


def item_deps(g: Geometry, kind: int, j: int) -> list[tuple[int, int]]:
    """Incoming (direction, neighbour step) chunks an item waits for."""
    if kind == ITEM_STAGE_F:
        st = send_step(g, FWD, j)
        return [(FWD, st.dep)] if st.src == SRC_RED else []
    if kind == ITEM_STAGE_B:
        st = send_step(g, BWD, j)
        return [(BWD, st.dep)] if st.src == SRC_RED else []
    if kind == ITEM_OUT_A:
        return [(FWD, 2 * g.na + g.nb + j)]
    if kind == ITEM_OUT_B:
        return [(BWD, g.nb + j), (FWD, fb_step(g, j))]
    if kind == ITEM_OUT_PREV:
        return [(FWD, j)]
    if kind == ITEM_OUT_NEXT:
        return [(BWD, j)]
    if kind == ITEM_OUT_FARF:
        return [(FWD, g.nc + j)]
    if kind == ITEM_OUT_FARB:
        return [(BWD, g.nc + j)]
    raise ValueError(kind)


def out_next_chunk(g: Geometry, s: int) -> int:
    """Block chunk carried by backward step s of the own-block send (second half first)."""
    return g.nh + s if s < g.nc - g.nh else s - (g.nc - g.nh)


# ------------------------------------------------------------------------------------------------ layout
def nc_max(max_rows: int, cr_max: int) -> int:
    q = max_rows // WORLD
    return 4 * math.ceil(math.ceil(q / cr_max) / 4)


@dataclass(frozen=True)
class Layout:
    """Byte layout of one rank's registered host region (the C side computes the same numbers)."""
    max_rows: int
    cr_max: int
    nslot: int

    def __post_init__(self):
        if self.max_rows % WORLD or not 0 < self.max_rows <= 1 << 16:
            raise ValueError("max_rows must be a positive multiple of 4")
        if not 1 <= self.cr_max <= 1024 or not 2 <= self.nslot <= 4096:
            raise ValueError("chunk rows in 1..1024 and 2..4096 staging slots")

    @property
    def steps_cap(self) -> int:
        """Upper bound of send steps per direction per call (RS backward: 2nc - 2*floor(nc/4) <= 1.5 nc + 1.5)."""
        return (3 * nc_max(self.max_rows, self.cr_max)) // 2 + 2

    @property
    def chunk_bytes(self) -> int:
        return self.cr_max * ROW_BYTES

    @property
    def dir_recv_bytes(self) -> int:
        return self.steps_cap * self.chunk_bytes

    # region = [recv][rflag][stage][ready][ctrl]
    def recv(self, parity: int, d: int) -> int:
        return (parity * DIRS + d) * self.dir_recv_bytes

    @property
    def rflag_off(self) -> int:
        return 2 * DIRS * self.dir_recv_bytes

    def rflag(self, parity: int, d: int, s: int) -> int:
        return self.rflag_off + ((parity * DIRS + d) * self.steps_cap + s) * FLAG_STRIDE

    @property
    def stage_off(self) -> int:
        return self.rflag_off + 2 * DIRS * self.steps_cap * FLAG_STRIDE

    def stage(self, d: int, slot: int) -> int:
        return self.stage_off + (d * self.nslot + slot) * self.chunk_bytes

    @property
    def ready_off(self) -> int:
        return self.stage_off + DIRS * self.nslot * self.chunk_bytes

    def ready(self, d: int, slot: int) -> int:
        """u64 = staged index + 1, written by the GPU once every part of the chunk is staged."""
        return self.ready_off + (d * self.nslot + slot) * FLAG_STRIDE

    @property
    def ctrl_off(self) -> int:
        return self.ready_off + DIRS * self.nslot * FLAG_STRIDE

    def send_done(self, d: int) -> int:
        """u64 = staged chunks of direction d whose RDMA write has completed (proxy -> GPU)."""
        return self.ctrl_off + d * FLAG_STRIDE

    @property
    def poison_off(self) -> int:
        return self.ctrl_off + 2 * FLAG_STRIDE

    @property
    def total(self) -> int:
        raw = self.ctrl_off + 4096
        return (raw + (2 << 20) - 1) // (2 << 20) * (2 << 20)


@dataclass
class Call:
    geom: Geometry
    seq: int            # 1, 2, ... shared by RS and AG calls, identical on every rank
    stage_base: tuple   # per direction: global staged index of this call's first staged chunk
    work_base: int      # device work counter at this call's first fetch

    @property
    def kind(self) -> int:
        return self.geom.kind

    @property
    def rows(self) -> int:
        return self.geom.rows

    @property
    def parity(self) -> int:
        return self.seq & 1


class Sequencer:
    """Host-side call numbering, identical on every rank because every rank issues the same calls in the same order."""

    def __init__(self, layout: Layout, grid: int, parts: int):
        self.layout, self.grid, self.parts = layout, grid, parts
        self.seq, self.stage_next, self.work_next = 0, [0, 0], 0

    def next(self, kind: int, rows: int) -> Call:
        if rows > self.layout.max_rows:
            raise ValueError("rows above the ring's maximum")
        g = geometry(kind, rows, self.layout.cr_max)
        assert max(g.steps(FWD), g.steps(BWD)) <= self.layout.steps_cap
        self.seq += 1
        if self.seq >= 1 << 31:
            raise RuntimeError("sequence space exhausted")
        call = Call(g, self.seq, tuple(self.stage_next), self.work_next)
        for d in (FWD, BWD):
            self.stage_next[d] += g.staged(d)
        self.work_next += g.items() * self.parts + self.grid       # every CTA fetches once past the end
        return call


# ------------------------------------------------------------------------------------------------ arithmetic
def bf16_round(x: float) -> float:
    """Round a float32-representable value to BF16, round-to-nearest-even (what __floats2bfloat162_rn does)."""
    import struct
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    if (bits & 0x7F800000) == 0x7F800000:
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000 | (0x400000 if bits & 0x7FFFFF else 0)))[0]
    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32(x: float) -> float:
    import struct
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_add(a: float, b: float) -> float:
    return bf16_round(f32(a + b))


def nccl_ring_rs_value(values, block: int) -> float:
    """NCCL's ring order for one element of block `block`: ((x[j+1] + x[j+2]) + x[j+3]) + x[j]."""
    j = block
    s = values[(j + 1) % WORLD]
    for k in (2, 3, 4):
        s = bf16_add(s, values[(j + k) % WORLD])
    return s
