"""GLM-5.3 prefill indexer row split: every TP rank scores its own share of a prefill chunk's query rows (opt-in).

Installed as ``glm53_indexer_rowsplit.py`` in site-packages; used only by the kpool indexer's prefill loop
(``vllm/model_executor/layers/sparse_attn_indexer_kpool.py``) when ``GLM53_INDEXER_ROW_SPLIT=1``. Unset or 0: the served
file never imports this module and every path is unchanged.

What changes. The kpool indexer is replicated across the TP ranks, so in prefill every rank computes the same FP8 MQA
logits, the same per-row top-k and the same canonical tie order for every query row of a chunk. With the switch on, a
chunk with at least ``GLM53_INDEXER_ROW_SPLIT_MIN_ROWS`` query rows and at least ``GLM53_INDEXER_ROW_SPLIT_MIN_POOLS``
compressed key positions is cut into contiguous, tile-aligned row ranges, one per rank. Each rank runs the unchanged
served kernels (logits, ``top_k_per_row_prefill``, ``canonicalize_topk``) on its rows only; the ranks then
all-gather the selected pool ids (int32 bytes, no floating-point collective). The pool expansion that follows runs
on the full, gathered rows exactly as before.

Why the result is the same bytes (by construction; the GPU leaf checks it on the served kernels): the logits of a row
depend only on that row's query, weights and key range; ``top_k_per_row_prefill`` and ``canonicalize_topk`` (lower index
wins ties, ascending output, fixed-order scans) work row by row on that row's scores and bounds; the per-row key
bounds (``cu_seqlen_ks`` / ``cu_seqlen_ke``) are sliced, never rebased, and the selected ids stay relative to each row's
start. The split decision is taken on the host from chunk metadata that is identical on every rank, and every rank
joins every gather (a rank without rows sends a -1 filled buffer of the common extent).

Scope: CUDA, FP8 indexer cache (``q_scale`` is None), ``index_kpool > 1``, eager prefill only (never while a CUDA graph
is being captured), no decode-context parallelism. Anything else keeps the served path.

Idea: the SG18 native prefill TP split of rhys101 (Rhys Jones), rhys101/DeepSeek-V4.1-Flash-vLLM-DGX-Spark-8, carried
into MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks (adapter/spark_prefill_dense.py) in the TP4 production line contributed by
knapcio. This module is an independent implementation for vLLM's GLM-5.3 kpool indexer; no code from either repository
is copied.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping

SELECTOR = "GLM53_INDEXER_ROW_SPLIT"
MIN_ROWS_ENV = "GLM53_INDEXER_ROW_SPLIT_MIN_ROWS"
MIN_POOLS_ENV = "GLM53_INDEXER_ROW_SPLIT_MIN_POOLS"
TILE_ENV = "GLM53_INDEXER_ROW_SPLIT_TILE"
CHECK_ENV = "GLM53_INDEXER_ROW_SPLIT_CHECK"
ENV_NAMES = (SELECTOR, MIN_ROWS_ENV, MIN_POOLS_ENV, TILE_ENV, CHECK_ENV)

DEFAULT_MIN_ROWS = 1024      # below this a chunk's indexer work is too small to pay for a gather
DEFAULT_MIN_POOLS = 4096     # 16,384 tokens of context at index_kpool 4
DEFAULT_TILE = 64            # rows; the busiest rank then stays within a few percent of a quarter at 128K-256K


@dataclass(frozen=True)
class Settings:
    enabled: bool
    min_rows: int
    min_pools: int
    tile: int
    check: bool

    def key(self) -> tuple:
        """What every rank must agree on (voted over the TP CPU group before the first split)."""
        return (self.enabled, self.min_rows, self.min_pools, self.tile, self.check)


def _int(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        raise ValueError(f"{name} must be a decimal integer, got {raw!r}")
    value = int(raw)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}, got {value}")
    return value


def parse_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    raw = env.get(SELECTOR, "").strip()
    if raw not in ("", "0", "1"):
        raise ValueError(f"{SELECTOR} must be unset, 0 or 1, got {raw!r}")
    check = env.get(CHECK_ENV, "").strip()
    if check not in ("", "0", "1"):
        raise ValueError(f"{CHECK_ENV} must be unset, 0 or 1, got {check!r}")
    tile = _int(env, TILE_ENV, DEFAULT_TILE, 16, 4096)
    if tile & (tile - 1):
        raise ValueError(f"{TILE_ENV} must be a power of two, got {tile}")
    min_rows = _int(env, MIN_ROWS_ENV, DEFAULT_MIN_ROWS, 1, 1 << 24)
    min_pools = _int(env, MIN_POOLS_ENV, DEFAULT_MIN_POOLS, 1, 1 << 24)
    return Settings(raw == "1", min_rows, min_pools, tile, check == "1")


def partition(rows: int, world: int, rank: int, tile: int = DEFAULT_TILE) -> tuple[int, int, int]:
    """Contiguous, tile-aligned row range ``[start, stop)`` of ``rank`` and the common collective extent.

    Every rank gets the same extent (a whole number of tiles), so the all-gather has equal shares; only the last
    non-empty rank can be short, and every rank after it is empty. Gathered rank-major, row ``i`` of the chunk is row
    ``i`` of the gathered buffer, so ``gathered[:rows]`` is the chunk in its original order.
    """
    for name, value in (("rows", rows), ("world", world), ("rank", rank), ("tile", tile)):
        if type(value) is not int:
            raise TypeError(f"{name} must be an int")
    if rows < 0 or world < 1 or not 0 <= rank < world or tile < 1:
        raise ValueError((rows, world, rank, tile))
    extent = -(-rows // (world * tile)) * tile
    start = min(rows, rank * extent)
    return start, min(rows, start + extent), extent


def engages(settings: Settings, *, rows: int, pools: int, index_kpool: int, world: int, capturing: bool,
            cuda: bool, fp8_cache: bool, local_pools: int | None = None) -> bool:
    """Host-side decision; all inputs are identical on every rank for the same chunk."""
    return bool(
        settings.enabled
        and cuda
        and fp8_cache
        and not capturing
        and index_kpool > 1
        and world > 1
        and rows >= settings.min_rows
        and pools >= settings.min_pools
        and (local_pools is None or local_pools == pools)   # no decode-context parallelism
    )


def split_rows(rows: int, world: int, rank: int, tile: int,
               local_select: Callable[[int, int, int], Any], gather: Callable[[Any], Any]) -> Any:
    """Run ``local_select(start, stop, extent)`` on this rank's rows and gather every rank's result.

    ``local_select`` returns an initialized ``[extent, K]`` buffer whose first ``stop - start`` rows hold this rank's
    selection (the rest stays -1). ``gather`` concatenates the equal-extent buffers of all ranks in rank order.
    """
    start, stop, extent = partition(rows, world, rank, tile)
    local = local_select(start, stop, extent)
    if len(local) != extent:
        raise RuntimeError(f"local selection has {len(local)} rows, expected the collective extent {extent}")
    full = gather(local)
    if len(full) != world * extent:
        raise RuntimeError(f"gathered {len(full)} rows, expected {world} x {extent}")
    return full[:rows]


class RowSplit:
    """Per-process runtime: the TP group, the voted settings and a few counters (torch is imported lazily)."""

    def __init__(self, settings: Settings, group: Any) -> None:
        self.settings = settings
        self.group = group
        self.rank = int(group.rank_in_group)
        self.world = int(group.world_size)
        self.engaged = 0
        self.check_calls = 0
        self.check_mismatched_rows = 0
        self._logged = False

    def plan(self, chunk: Any, *, index_kpool: int, fp8_cache: bool) -> bool:
        import torch

        rows = int(chunk.token_end - chunk.token_start)
        pools = int(chunk.total_seq_lens)
        local_pools = getattr(chunk, "local_total_seq_lens", None)
        return engages(self.settings, rows=rows, pools=pools, index_kpool=index_kpool, world=self.world,
                       capturing=torch.cuda.is_current_stream_capturing(), cuda=torch.cuda.is_available(),
                       fp8_cache=fp8_cache, local_pools=None if local_pools is None else int(local_pools))

    def select(self, *, rows: int, select_k: int, device: Any, starts: Any, ends: Any,
               logits: Callable[[int, int], Any], topk: Callable[..., Any], canon: Callable[..., Any]) -> Any:
        """``[rows, select_k]`` int32 pool ids of the whole chunk, identical on every rank.

        ``logits(s, e)`` returns the fp32 logits of chunk rows ``[s, e)`` (the served scorer on the sliced query,
        weights and bounds); ``topk`` is ``torch.ops._C.top_k_per_row_prefill``; ``canon`` is ``canonicalize_topk``.
        """
        import torch

        def local_select(start: int, stop: int, extent: int) -> Any:
            buf = torch.full((extent, select_k), -1, dtype=torch.int32, device=device)
            n = stop - start
            if n > 0:
                scores = logits(start, stop)
                out = buf[:n]
                topk(scores, starts[start:stop], ends[start:stop], out, n, scores.stride(0), scores.stride(1),
                     select_k)
                canon(scores, starts[start:stop], ends[start:stop], out)
            return buf

        def gather(local: Any) -> Any:
            return self.group.all_gather(local.contiguous(), dim=0)

        result = split_rows(rows, self.world, self.rank, self.settings.tile, local_select, gather)
        self.engaged += 1
        if not self._logged:
            self._logged = True
            start, stop, extent = partition(rows, self.world, self.rank, self.settings.tile)
            print(f"GLM53_INDEXER_ROW_SPLIT engaged: rank {self.rank}/{self.world} rows {rows} -> [{start}, {stop}) "
                  f"extent {extent}, select_k {select_k}, min_rows {self.settings.min_rows}, "
                  f"min_pools {self.settings.min_pools}, tile {self.settings.tile}", file=sys.stderr, flush=True)
        if self.settings.check:
            # Diagnostic only: the served full-row selection on this rank, compared byte for byte.
            full = local_select(0, rows, rows)
            differing = int((full != result).any(dim=1).sum().item())
            self.check_calls += 1
            self.check_mismatched_rows += differing
            if differing or self.check_calls <= 4:
                print(f"GLM53_INDEXER_ROW_SPLIT_CHECK rank {self.rank} call {self.check_calls} rows {rows} "
                      f"differing_rows {differing} (total {self.check_mismatched_rows})", file=sys.stderr, flush=True)
        return result


_RUNTIME: list = []


def vote(settings: Settings, group: Any) -> None:
    """All TP ranks must run the same settings: gather them over the CPU group once, raise on any difference."""
    import torch.distributed as dist

    mine = settings.key()
    items: list = [None] * int(group.world_size)
    dist.all_gather_object(items, mine, group=group.cpu_group)
    if any(item != mine for item in items):
        raise RuntimeError(f"{SELECTOR} settings differ across TP ranks: {items}")


def runtime() -> RowSplit:
    """The process's RowSplit, built (and voted) on first use; call only with the selector on."""
    if not _RUNTIME:
        from vllm.distributed import get_tp_group

        settings = parse_settings()
        if not settings.enabled:
            raise RuntimeError(f"{SELECTOR} runtime requested with the selector off")
        group = get_tp_group()
        vote(settings, group)
        _RUNTIME.append(RowSplit(settings, group))
        print(f"GLM53_INDEXER_ROW_SPLIT ready: rank {group.rank_in_group}/{group.world_size} settings "
              f"{settings.key()}", file=sys.stderr, flush=True)
    return _RUNTIME[0]
