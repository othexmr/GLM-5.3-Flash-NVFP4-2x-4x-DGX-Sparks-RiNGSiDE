"""Tile-major NVFP4 expert-weight layout for the GLM-5.3 TP4 dynamic MoE kernel.

Mechanism: the pinned kernel streams every FC1/FC2 weight tile with one TMA box
of 128 rows x 64 bytes taken from row-major expert matrices, so each 8 KB box
is 128 separate 64-byte segments (2048-byte stride for W13, 256-byte stride for
W2).  This module defines a permutation of the packed FP4 payload bytes only,
so that every box the kernel issues is one contiguous 8 KB range and the boxes
of one task follow each other in the kernel's own consumption order.  Block
scales, global scales, activations, the MMA operand bytes delivered to shared
memory and the arithmetic are unchanged; the output contract is bit-identical.

The functions here are pure index arithmetic (usable without torch) plus a
torch repack.  They are copied beside the pinned B12X overlay by
build_overlay.py as ``b12x/moe/fused_moe/_tp4_tile_major.py``.
"""

TILE_ROWS = 128          # TMA box rows (tile_n) of the pinned kernel
TILE_BYTES = 64          # TMA box row bytes: tile_k = 128 FP4 elements
BOX_BYTES = TILE_ROWS * TILE_BYTES   # 8192


class Geometry:
    """Local per-rank expert geometry (TP4 GLM-5.3-Flash NVFP4 by default)."""

    def __init__(self, hidden=4096, intermediate=512, experts=288):
        if hidden % TILE_ROWS or intermediate % TILE_ROWS or hidden % 256:
            raise ValueError('hidden and intermediate must be multiples of 128')
        self.k = int(hidden); self.n = int(intermediate); self.e = int(experts)
        self.kt = self.k // 128             # FC1 k-tiles per row (32)
        self.g = self.n // TILE_ROWS        # intermediate slices / gate tiles (4)
        self.ot = self.k // TILE_ROWS       # FC2 output tiles (32)
        self.st = self.n // 128             # FC2 k-tiles = slices (4)
        self.rows13 = 2 * self.n            # stock W13 rows [up; gate] (1024)
        self.tiles13 = 2 * self.g * self.kt # W13 boxes per expert (256)
        self.tiles2 = self.st * self.ot     # W2 boxes per expert (128)
        self.tm_rows13 = self.tiles13 * TILE_ROWS   # 32768
        self.tm_rows2 = self.tiles2 * TILE_ROWS     # 16384

    def as_dict(self):
        return {'hidden': self.k, 'intermediate': self.n, 'experts': self.e,
                'k_tiles': self.kt, 'gate_tiles': self.g, 'output_tiles': self.ot,
                'w13_tiles_per_expert': self.tiles13, 'w2_tiles_per_expert': self.tiles2,
                'tile_major_w13_rows': self.tm_rows13, 'tile_major_w2_rows': self.tm_rows2}


# --- index arithmetic (the single source of truth for host repack and kernel) --

def w13_tile(geometry, n_tile, k_tile):
    """Tile-major box index of stock W13 N-tile ``n_tile`` (0..2g-1; up tiles
    first, gate tiles from g) and k-tile ``k_tile``.  Order: for slice s:
    gate(s) k 0..kt-1, then up(s) k 0..kt-1 (the kernel's consumption order)."""
    g, kt = geometry.g, geometry.kt
    if not 0 <= n_tile < 2 * g or not 0 <= k_tile < kt: raise ValueError('W13 tile out of range')
    s = n_tile % g
    pass_index = 0 if n_tile >= g else 1        # gate pass first, then up pass
    return (s * 2 + pass_index) * kt + k_tile


def w2_tile(geometry, out_tile, slice_index):
    """Tile-major box index of stock W2 output tile ``out_tile`` (hidden rows
    out_tile*128..) at intermediate k-tile ``slice_index``: slice-major."""
    if not 0 <= out_tile < geometry.ot or not 0 <= slice_index < geometry.st: raise ValueError('W2 tile out of range')
    return slice_index * geometry.ot + out_tile


def kernel_bases(geometry, slice_index):
    """The three per-slice box bases the kernel adds k_tile / output_tile to."""
    kt, ot = geometry.kt, geometry.ot
    return {'gate': slice_index * 2 * kt, 'up': slice_index * 2 * kt + kt, 'down': slice_index * ot}


def w13_permutation(geometry):
    """perm[t] = stock (n_tile, k_tile) flattened index (n_tile*kt + k_tile) whose
    8 KB box becomes tile-major box t."""
    g, kt = geometry.g, geometry.kt
    perm = [None] * geometry.tiles13
    for n_tile in range(2 * g):
        for k_tile in range(kt):
            perm[w13_tile(geometry, n_tile, k_tile)] = n_tile * kt + k_tile
    if any(p is None for p in perm): raise AssertionError('W13 permutation incomplete')
    return perm


def w2_permutation(geometry):
    """perm[t] = stock (out_tile, slice) flattened index (out_tile*st + slice)."""
    ot, st = geometry.ot, geometry.st
    perm = [None] * geometry.tiles2
    for out_tile in range(ot):
        for s in range(st):
            perm[w2_tile(geometry, out_tile, s)] = out_tile * st + s
    if any(p is None for p in perm): raise AssertionError('W2 permutation incomplete')
    return perm


# --- byte-address model (tests and the read-pattern diagnostic) ----------------

def stock_w13_box(geometry, n_tile, k_tile):
    """Byte ranges (offset, length) read by one stock FC1 box within an expert."""
    row_bytes = geometry.k // 2
    base = n_tile * TILE_ROWS * row_bytes + k_tile * TILE_BYTES
    return [(base + r * row_bytes, TILE_BYTES) for r in range(TILE_ROWS)]


def stock_w2_box(geometry, out_tile, slice_index):
    row_bytes = geometry.n // 2
    base = out_tile * TILE_ROWS * row_bytes + slice_index * TILE_BYTES
    return [(base + r * row_bytes, TILE_BYTES) for r in range(TILE_ROWS)]


def tile_major_box(tile_index):
    return [(tile_index * BOX_BYTES, BOX_BYTES)]


def task_sequence(geometry):
    """Box sequence one M-tile task issues (deterministic path: all slices):
    list of ('w13'|'w2', stock coordinates, tile-major index)."""
    seq = []
    for s in range(geometry.g):
        for k in range(geometry.kt): seq.append(('w13', (geometry.g + s, k), w13_tile(geometry, geometry.g + s, k)))
        for k in range(geometry.kt): seq.append(('w13', (s, k), w13_tile(geometry, s, k)))
        for o in range(geometry.ot): seq.append(('w2', (o, s), w2_tile(geometry, o, s)))
    return seq


# --- numpy reference ---------------------------------------------------------

def repack_w13_numpy(w13_u8, geometry):
    import numpy as np
    e, rows, half_k = w13_u8.shape
    if (rows, half_k) != (geometry.rows13, geometry.k // 2): raise ValueError('stock W13 shape mismatch')
    boxes = w13_u8.reshape(e, 2 * geometry.g, TILE_ROWS, geometry.kt, TILE_BYTES).transpose(0, 1, 3, 2, 4)
    boxes = boxes.reshape(e, geometry.tiles13, TILE_ROWS, TILE_BYTES)
    return np.ascontiguousarray(boxes[:, np.asarray(w13_permutation(geometry))]).reshape(e, geometry.tm_rows13, TILE_BYTES)


def repack_w2_numpy(w2_u8, geometry):
    import numpy as np
    e, rows, half_n = w2_u8.shape
    if (rows, half_n) != (geometry.k, geometry.n // 2): raise ValueError('stock W2 shape mismatch')
    boxes = w2_u8.reshape(e, geometry.ot, TILE_ROWS, geometry.st, TILE_BYTES).transpose(0, 1, 3, 2, 4)
    boxes = boxes.reshape(e, geometry.tiles2, TILE_ROWS, TILE_BYTES)
    return np.ascontiguousarray(boxes[:, np.asarray(w2_permutation(geometry))]).reshape(e, geometry.tm_rows2, TILE_BYTES)


def unpack_w13_numpy(tm_u8, geometry):
    import numpy as np
    e = tm_u8.shape[0]
    boxes = tm_u8.reshape(e, geometry.tiles13, TILE_ROWS, TILE_BYTES)
    inverse = np.argsort(np.asarray(w13_permutation(geometry)))
    stock = boxes[:, inverse].reshape(e, 2 * geometry.g, geometry.kt, TILE_ROWS, TILE_BYTES).transpose(0, 1, 3, 2, 4)
    return np.ascontiguousarray(stock).reshape(e, geometry.rows13, geometry.k // 2)


# --- torch repack (device or host tensors) -------------------------------------

def _permute_boxes_torch(tensor_u8, dims, perm, out_rows):
    """Generic torch box permutation. ``dims`` = (outer_tiles, rows, inner_tiles)."""
    import torch
    e = tensor_u8.shape[0]
    outer, inner = dims
    boxes = tensor_u8.reshape(e, outer, TILE_ROWS, inner, TILE_BYTES).permute(0, 1, 3, 2, 4)
    boxes = boxes.reshape(e, outer * inner, TILE_ROWS, TILE_BYTES)
    index = torch.as_tensor(perm, dtype=torch.long, device=tensor_u8.device)
    return boxes.index_select(1, index).reshape(e, out_rows, TILE_BYTES).contiguous()


def repack_w13_torch(w13_u8, geometry):
    if tuple(w13_u8.shape[1:]) != (geometry.rows13, geometry.k // 2): raise ValueError('stock W13 shape mismatch')
    return _permute_boxes_torch(w13_u8, (2 * geometry.g, geometry.kt), w13_permutation(geometry), geometry.tm_rows13)


def repack_w2_torch(w2_u8, geometry):
    if tuple(w2_u8.shape[1:]) != (geometry.k, geometry.n // 2): raise ValueError('stock W2 shape mismatch')
    return _permute_boxes_torch(w2_u8, (geometry.ot, geometry.st), w2_permutation(geometry), geometry.tm_rows2)


def repack_inplace_torch(storage_u8, repack, geometry, chunk_bytes=64 << 20):
    """Rewrite ``storage_u8`` [E, rows, cols] bytes into tile-major order in
    place, a bounded chunk of experts at a time (the load-time serving path;
    no model-sized duplicate).  Transient allocation per chunk is about three
    times ``chunk_bytes`` (the chunk clone, the box reshape copy and the
    index_select output), i.e. roughly 192 MiB at the 64 MiB default.  This is
    a live-tensor estimate, not a measured CUDA allocator/reserved-memory peak.
    The previous chunk's output is released before the next chunk allocates.
    Returns a [E, tm_rows, 64] view of the storage."""
    import torch
    if not storage_u8.is_contiguous() or storage_u8.dtype != torch.uint8: raise ValueError('contiguous uint8 storage required')
    e = storage_u8.shape[0]; per_expert = storage_u8[0].numel()
    step = max(1, min(e, chunk_bytes // per_expert))
    flat = storage_u8.view(e, per_expert)
    for e0 in range(0, e, step):
        block = storage_u8[e0:e0 + step]
        packed = repack(block.clone(), geometry)        # temporary: one chunk
        flat[e0:e0 + step].copy_(packed.reshape(packed.shape[0], per_expert))
        del packed  # do not retain the previous output during the next chunk's allocations
    tm_rows = per_expert // TILE_BYTES
    return storage_u8.view(e, tm_rows, TILE_BYTES)
