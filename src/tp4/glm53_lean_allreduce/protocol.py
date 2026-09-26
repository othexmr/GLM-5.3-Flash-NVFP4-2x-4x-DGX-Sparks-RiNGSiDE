"""Pure TP4 physical topology and reduction-order contract."""

WORLD = 4
# Local HCA indices into ("rocep1s0f0", "rocep1s0f1", "roceP2p1s0f0", "roceP2p1s0f1") per directed edge
# (sender, receiver). Measured on the ring 2026-09-23 02:10 from each node's RoCE v2 GID table (index 3) and
# `ip -4 addr` (run-1 timed out at RTR with the earlier assumed f0->next/f1->prev map): every cable joins the
# SAME function index at both ends --
#   0<->1 10.100.224.0/24 f1 | 1<->2 10.100.225.0/24 f0 | 2<->3 10.100.226.0/24 f1 | 3<->0 10.100.227.0/24 f0
# with the two PCIe-domain PFs of that port as the two lanes (lane i pairs HCA i with the peer's HCA i).
DIRECT_PATHS = {
    (0, 1): (1, 3), (1, 0): (1, 3),
    (1, 2): (0, 2), (2, 1): (0, 2),
    (2, 3): (1, 3), (3, 2): (1, 3),
    (3, 0): (0, 2), (0, 3): (0, 2),
}


def peer(rank: int, phase: int) -> int:
    if rank not in range(WORLD) or phase not in (1, 2):
        raise ValueError("rank must be 0..3 and phase 1 or 2")
    return rank ^ 1 if phase == 1 else 3 - rank


def pair_local_rank(rank: int, phase: int) -> int:
    peer(rank, phase)
    return rank & 1 if phase == 1 else (0 if rank < 2 else 1)


def pair_map(rank: int, phase: int) -> tuple[tuple[int, ...], ...]:
    local = pair_local_rank(rank, phase)
    paths = DIRECT_PATHS[(rank, peer(rank, phase))]
    return ((), paths) if local == 0 else (paths, ())


def canonical_reduce(values: tuple[float, float, float, float], rounder=float) -> float:
    """CPU model of the two rounded pair adds, independent of observing rank."""
    pair_a = rounder(values[0] + values[1])
    pair_b = rounder(values[2] + values[3])
    return rounder(pair_a + pair_b)
