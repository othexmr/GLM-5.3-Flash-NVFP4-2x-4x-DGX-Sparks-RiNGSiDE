"""Replay-boundary checkpoints for GLM-5.3-Flash on DFlash2 stacks (replay-boundary lane, 2026-09-24;
outputs/2026-09-24-replay-boundary/DESIGN.md).

A prefix hit on this hybrid stack must be supported by the MLA blocks, the four KDA state groups and the DFlash2
drafter's sliding-window blocks at one token count. The drafter's EAGLE peek (the block past the hit must match) caps a
replay of a P-token prompt at R(P) = floor((P - 1) / H) * H - H (H = the hash unit = the drafter block, 1,152), and an
extension that keeps the whole prompt reaches at least R(P). With GLM53_REPLAY_BOUNDARY=1 the scheduler moves its
prompt-boundary stop from the prompt's last hash boundary to R(P) and the managers keep one complete hybrid state there:
the KDA state (a chunk-end state CoW-copied by the image's partial-tail path, or at TP4 an in-chunk export), the drafter
blocks around it and an MLA partial entry. This module holds the pure policy (R, the stop, the TP4 export plan) and the
scheduler-side registry that bounds, ages and validates the retained KDA boundary states.

Everything here runs in the scheduler process (rank 0's engine core). Default off: `enable()` returns False and no
changed code path runs.

v2 (2026-09-24 20:1x, after the v1 boot refusal of chain 274): `enable()` picks the lever's groups among the
prefix-cacheable KV cache groups only, as the image's HybridKVCacheCoordinator does for hit lookup. The real GLM-5.3
scheduler layout (`_get_kv_cache_groups_glm5_next`) has seven groups: MLA+indexer, the kpool tail (KpoolTailSpec, a
SlidingWindowSpec subclass with prefix_cacheable False, per-request scratch managed by KpoolTailManager ->
CircularBufferManager -> FullAttentionManager), four KDA (mamba align) groups and the DFlash2 drafter (SlidingWindowSpec,
is_eagle_group). v1 counted the tail as a second sliding-window group and refused the boot.

admit fix (main session, 2026-09-25 01:4x, on 144c7d34 = v2): the final TP4 run (chain 287) logged five
`GLM53_REPLAY_BOUNDARY invalid` at boundary 1,152 during tool-eval-bench, none during RigMark. Cause: the image's own
partial tail. A prompt of 1,153-2,304 tokens caches its KDA state at its last hash boundary (1,152) through
`_cache_partial_tail_block` without this registry; a 2,305-3,456-token prompt with the same first 1,152 tokens records
the replay boundary under the same hash. The pool keeps both blocks under that key and returns the first one (its order
changes with inserts and CoW moves), so a lookup could get the image's block, which the registry had never recorded:
v2 called it stale, refused a valid hit and dropped every block under the hash, the image's own included. Now the
registry judges only blocks it recorded for the entry; a block it never recorded is the image's own entry for the same
prefix (the pool's hash map is its proof, as for any native hit) and is accepted. An invalidation or a cap eviction
removes the entry's recorded blocks from the pool (a stale tag included) and leaves the image's own entries alone.
"""

from __future__ import annotations

import os
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

ENABLED = os.environ.get("GLM53_REPLAY_BOUNDARY", "0") == "1"
MAX_BOUNDARIES = int(os.environ.get("GLM53_REPLAY_BOUNDARY_MAX", "16"))
CHUNK_TOKENS = 16          # b12x KDA prefill tile: in-chunk export offsets are multiples of it
BOOT_ID = uuid.uuid4().hex[:12]

_LOG = None


def _log():
    global _LOG
    if _LOG is None:
        try:
            from vllm.logger import init_logger
            _LOG = init_logger("vllm.glm53_replay_boundary")
        except Exception:  # pragma: no cover  (CPU tests without vLLM)
            import logging
            _LOG = logging.getLogger("glm53_replay_boundary")
    return _LOG


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def replay_boundary(num_prompt_tokens: int, unit: int) -> int:
    """R(P): the longest prefix a replay of a ``num_prompt_tokens`` prompt can hit when the EAGLE-group drafter's block
    is ``unit`` tokens: the replay's max hit length is P - 1 and the drafter needs the block past the hit inside it."""
    if unit <= 0:
        raise ValueError(f"unit {unit} must be positive")
    return max(0, (num_prompt_tokens - 1) // unit * unit - unit)


@dataclass(frozen=True)
class Boundary:
    """The lever's decision for one request in one scheduling pass: ``tokens`` = R(P); ``stop`` = R(P) when it is a new
    chunk stop (last_cache_position < R(P) < P), else 0 (no stop, but the relaxed mid-block restart still applies)."""
    tokens: int
    stop: int


def request_boundary(scheduler, request, block_size: int, last_cache_position: int,
                     num_external_computed_tokens: int) -> Boundary | None:
    """Called from Scheduler._mamba_block_aligned_split. None = the lever does not apply to this request (the split
    is exactly today's). Applies to pure prompt prefill (no outputs, no connector tokens, no encoder inputs) on the
    retained-checkpoint path."""
    if not getattr(scheduler, "_glm53_rb", False):
        return None
    if (
        block_size != scheduler.block_size
        or num_external_computed_tokens
        or request.num_output_tokens != 0
        or request.num_tokens != request.num_prompt_tokens
        or request.has_encoder_inputs
    ):
        return None
    unit = scheduler.hash_block_size
    tokens = replay_boundary(request.num_prompt_tokens, unit)
    if tokens <= 0:
        return None
    # the managers read it while hashing (drafter retention, MLA partial entry, KDA partial tail)
    request.glm53_replay_boundary = tokens
    stop = tokens if last_cache_position < tokens < request.num_prompt_tokens else 0
    return Boundary(tokens=tokens, stop=stop)


def plan_chunk(start: int, end: int, stops, grid: int, boundary: int, max_exports: int) -> tuple[int, tuple[int, ...]]:
    """TP4 (in-chunk KDA exports): kda-checkpoints' plan_chunk with the replay boundary. Stops are walked in order; a
    stop on the ``grid`` (the mamba block) or equal to ``boundary`` that is 16-aligned from ``start`` becomes an export
    (at most ``max_exports``); the first other stop ends the chunk. Export columns (``cdiv(p, grid) - 1``: an on-grid
    state in its own column, the off-grid boundary in the column of the block that holds it) must be distinct and must
    not be the column of the chunk's final state; if the boundary breaks that, the chunk ends at the boundary instead
    (the chunk-end path). Without an exportable stop the result is the scheduler's own ``min(stops inside)``."""
    if grid <= 0 or grid % CHUNK_TOKENS:
        raise ValueError(f"grid {grid} must be a positive multiple of {CHUNK_TOKENS}")
    exports: list[int] = []
    stop_at = end
    for stop in sorted({int(s) for s in stops if start < s < end}):
        on_grid = stop % grid == 0
        if (on_grid or stop == boundary) and (stop - start) % CHUNK_TOKENS == 0 and len(exports) < max_exports:
            exports.append(stop)
            continue
        stop_at = stop
        break
    final_col = cdiv(stop_at, grid) - 1
    cols = [cdiv(p, grid) - 1 for p in exports]
    if boundary in exports:
        b_col = cdiv(boundary, grid) - 1
        clash = b_col == final_col or cols.count(b_col) > 1
        if clash:
            exports = [p for p in exports if p < boundary]
            stop_at = boundary
    return stop_at, tuple(exports)


# ---------------------------------------------------------------------------------------------------- registry
@dataclass
class Entry:
    key: bytes                      # the prefix-chain block hash at ``tokens`` (without the group id)
    tokens: int
    request_id: str                 # the first producer
    path: str                       # "chunk-end" (CoW at the next step) or "export" (TP4 in-chunk)
    blocks: dict = field(default_factory=dict)   # block id -> (group id, KVCacheBlock, epoch)
    created: float = 0.0
    last_used: float = 0.0

    def groups(self) -> set:
        return {gid for gid, _, _ in self.blocks.values()}


class Registry:
    """The live KDA boundary states. Each KDA group's manager records the block it registered under the boundary's
    hash (one entry per hash; two producers of the same prompt add their blocks to the same entry). A boundary is
    complete when every KDA group holds it. Bounded (``cap`` entries, least recently used evicted), validated before a
    hit is admitted: every returned block this registry recorded for the entry must still be that group's block,
    tagged with the same (epoch, hash, tokens) and mapped by the pool under the hash. A recorded block that fails this
    is stale: the entry's recorded blocks leave the pool's map and the lookup continues below it, so the request
    recomputes. A returned block the registry never recorded for the entry is the image's own partial tail or
    checkpoint of the same prefix and is accepted (admit fix)."""

    def __init__(self, cap: int, group_ids, pool, make_key):
        self.cap, self.pool, self.make_key = int(cap), pool, make_key
        self.group_ids = tuple(group_ids)
        self.num_groups = len(self.group_ids)
        self.entries: OrderedDict[bytes, Entry] = OrderedDict()
        # block id -> (epoch, key, tokens) of its registration (KVCacheBlock has __slots__: the tag lives here)
        self.tags: dict[int, tuple[int, bytes, int]] = {}
        self.epoch = 0
        self.counts = dict(captures=0, complete=0, admits=0, invalid=0, evicted=0, dead=0, moved=0, native=0)
        self._marks: dict[str, int] = {}

    def _mark(self, kind: str, text: str) -> None:
        n = self._marks[kind] = self._marks.get(kind, 0) + 1
        if n <= 8 or n & (n - 1) == 0:
            _log().info("GLM53_REPLAY_BOUNDARY %s %d: %s | %s", kind, n, text,
                        " ".join(f"{k}={v}" for k, v in self.stats().items()))

    def record(self, key: bytes, tokens: int, request_id: str, group_id: int, block, path: str) -> None:
        """A KDA group registered ``block`` under ``key`` (a partial entry at ``tokens``) for ``request_id``."""
        now = time.monotonic()
        e = self.entries.get(key)
        if e is not None and e.tokens != tokens:     # cannot happen (the hash names the prefix); be strict anyway
            self._drop(e)
            e = None
        if e is None:
            e = Entry(key=key, tokens=tokens, request_id=request_id, path=path, created=now)
            self.entries[key] = e
            self.counts["captures"] += 1
        was_complete = len(e.groups()) == self.num_groups
        self.epoch += 1
        e.blocks[block.block_id] = (group_id, block, self.epoch)
        self.tags[block.block_id] = (self.epoch, key, tokens)
        e.last_used = now
        self.entries.move_to_end(key)
        if not was_complete and len(e.groups()) == self.num_groups:
            self.counts["complete"] += 1
            self._mark("capture", f"request {request_id} boundary {tokens} path {path} boot {BOOT_ID}")
        self._enforce_cap(keep=key)

    def moved(self, src, dst, group_id: int) -> None:
        """The image's CoW moved a partial entry from ``src`` (the producer's running slot) to ``dst``."""
        tag = self.tags.get(src.block_id)
        if tag is None:
            return
        e = self.entries.get(tag[1])
        rec = e.blocks.get(src.block_id) if e is not None else None
        if rec is None or rec[0] != group_id or rec[1] is not src or rec[2] != tag[0]:
            return
        del e.blocks[src.block_id]
        e.blocks[dst.block_id] = (group_id, dst, tag[0])
        self.tags[dst.block_id] = tag
        del self.tags[src.block_id]
        self.counts["moved"] += 1

    def _valid(self, e: Entry, group_id: int, block) -> bool:
        rec = e.blocks.get(block.block_id)
        return (
            rec is not None
            and rec[0] == group_id
            and rec[1] is block
            and not block.is_null
            and self.tags.get(block.block_id) == (rec[2], e.key, e.tokens)
            and self.pool.cached_block_hash_to_block.contain(self.make_key(e.key, group_id), block.block_id)
        )

    def admit(self, key: bytes, blocks, group_ids) -> bool:
        """Before a KDA hit at ``key`` is used: True if ``key`` is not a registered boundary (not ours to judge) or
        every returned block the registry recorded for it is still valid; a returned block it never recorded for the
        entry is the image's own entry for the same prefix and is accepted (admit fix). False after invalidating the
        entry's recorded blocks."""
        e = self.entries.get(key)
        if e is None:
            return True
        judged = [(gid, blk) for gid, blk in zip(group_ids, blocks) if blk.block_id in e.blocks]
        if not all(self._valid(e, gid, blk) for gid, blk in judged):
            self.counts["invalid"] += 1
            self._drop(e)
            self._mark("invalid", f"boundary {e.tokens} request {e.request_id}")
            return False
        if len(judged) < len(group_ids):
            self.counts["native"] += 1
            self._mark("native", f"boundary {e.tokens}: {len(group_ids) - len(judged)} of {len(group_ids)} groups "
                                 f"hit the image's own partial tail")
            if not judged:
                return True
        e.last_used = time.monotonic()
        self.entries.move_to_end(key)
        self.counts["admits"] += 1
        self._mark("admit", f"boundary {e.tokens}")
        return True

    def _drop(self, e: Entry) -> None:
        """Invalidate the boundary: in every KDA group, each block the registry recorded for it that the pool still
        maps under its hash loses that entry (a block whose primary entry it is is uncached as eviction would uncache
        it), whatever its tag says; blocks the image registered under the same hash itself are left alone (admit fix).
        Then forget it."""
        cache = self.pool.cached_block_hash_to_block
        for gid, blk, epoch in list(e.blocks.values()):
            key_gid = self.make_key(e.key, gid)
            if cache.contain(key_gid, blk.block_id):
                if blk.block_hash == key_gid:
                    self.pool._remove_cached_block_hashes(blk)
                else:
                    cache.pop(key_gid, blk.block_id)
                    extra = self.pool.cached_block_hashes_by_block.get(blk.block_id)
                    if extra is not None:
                        extra.discard(key_gid)
                        if not extra:
                            self.pool.cached_block_hashes_by_block.pop(blk.block_id, None)
                if cache.contain(key_gid, blk.block_id):      # pragma: no cover  (never leave a stale entry)
                    raise RuntimeError(f"replay-boundary: cannot drop entry {key_gid!r} of block {blk.block_id}")
            if self.tags.get(blk.block_id) == (epoch, e.key, e.tokens):
                del self.tags[blk.block_id]
        self.entries.pop(e.key, None)

    def _dead(self, e: Entry) -> bool:
        """No block of the entry is still hittable (the pool evicted them), or a group lost its only block."""
        live = {gid for gid, blk, _ in e.blocks.values() if self._valid(e, gid, blk)}
        return len(live) < self.num_groups

    def _enforce_cap(self, keep: bytes) -> None:
        if len(self.entries) <= self.cap:
            return
        for e in list(self.entries.values()):          # dead entries first, oldest first
            if len(self.entries) <= self.cap:
                return
            if e.key != keep and self._dead(e):
                self.counts["dead"] += 1
                self._drop(e)
        while len(self.entries) > self.cap:               # then least recently used
            e = next(iter(self.entries.values()))
            if e.key == keep:
                self.entries.move_to_end(keep)
                e = next(iter(self.entries.values()))
                if e.key == keep:
                    break
            self.counts["evicted"] += 1
            self._drop(e)
            self._mark("evict", f"boundary {e.tokens} request {e.request_id}")

    def clear(self) -> None:
        for e in list(self.entries.values()):
            self._drop(e)

    def stats(self) -> dict:
        return dict(self.counts, live=len(self.entries), cap=self.cap)


REGISTRY: Registry | None = None


def registry() -> Registry | None:
    return REGISTRY


def enable(scheduler) -> bool:
    """Scheduler startup. GLM53_REPLAY_BOUNDARY=1 on an unsupported geometry fails the boot instead of running
    silently without the feature. Installs the registry and the managers' flags; logs the resolved geometry."""
    global REGISTRY
    if not ENABLED:
        return False
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager, MambaManager, SlidingWindowManager)
    from vllm.v1.kv_cache_interface import MambaSpec, SlidingWindowSpec

    problems = []
    coord = scheduler.kv_cache_manager.coordinator
    groups = scheduler.kv_cache_config.kv_cache_groups
    managers = coord.single_type_managers
    if len(managers) != len(groups):
        problems.append(f"{len(managers)} managers for {len(groups)} KV cache groups")
    # v2: only prefix-cacheable groups take part in hits (the image's coordinator skips the others, e.g. the kpool tail
    # scratch group: KpoolTailSpec is a SlidingWindowSpec subclass with prefix_cacheable False, KpoolTailManager is a
    # FullAttentionManager subclass); the lever never touches a non-cacheable group
    cacheable = [(m, g) for m, g in zip(managers, groups) if getattr(g.kv_cache_spec, "prefix_cacheable", True)]
    scratch = [m for m, g in zip(managers, groups) if not getattr(g.kv_cache_spec, "prefix_cacheable", True)]
    if not getattr(scheduler, "need_mamba_block_aligned_split", False):
        problems.append("needs the mamba align block split")
    if not getattr(scheduler, "mamba_partial_cache_hit", False) or not getattr(coord, "enable_partial_hash_hits", False):
        problems.append("needs fine-grained partial hash hits (hash block < mamba block)")
    if not getattr(scheduler, "use_eagle", False):
        problems.append("needs an EAGLE-group drafter (the boundary is the drafter's replay limit)")
    if getattr(scheduler.scheduler_config, "async_scheduling", False) or type(scheduler).__name__ == "AsyncScheduler":
        problems.append("needs --no-async-scheduling (the capture fence is one step)")
    if getattr(scheduler, "connector", None) is not None:
        problems.append("KV connectors are not supported")
    mamba = [m for m, g in cacheable if isinstance(g.kv_cache_spec, MambaSpec)]
    swa = [(m, g) for m, g in cacheable if isinstance(g.kv_cache_spec, SlidingWindowSpec)]
    fa = [m for m, g in cacheable if type(m) is FullAttentionManager]
    if not fa:
        problems.append("needs a full-attention (MLA) group")
    if not mamba or not all(isinstance(m, MambaManager) and m.mamba_cache_mode == "align" for m in mamba):
        problems.append("needs mamba 'align' groups")
    if any(m.num_speculative_blocks for m in mamba):
        problems.append("needs 0 speculative mamba blocks (RecoverSSM)")
    if any(g.kv_cache_spec.num_prefill_checkpoint_blocks for m, g in cacheable
           if isinstance(g.kv_cache_spec, MambaSpec)):
        problems.append("vLLM prefill-checkpoint blocks are not supported")
    if len(swa) != 1 or not isinstance(swa[0][0], SlidingWindowManager) or not swa[0][0].use_eagle:
        problems.append(f"needs exactly one prefix-cacheable EAGLE sliding-window drafter group (found {len(swa)})")
    elif swa[0][0].block_size != scheduler.hash_block_size:
        problems.append(f"drafter block {swa[0][0].block_size} != hash block {scheduler.hash_block_size}")
    if len({m.block_size for m in mamba}) != 1 or mamba[0].block_size != scheduler.block_size:
        problems.append("mamba block must equal the scheduler block")
    if problems:
        raise ValueError("GLM53_REPLAY_BOUNDARY=1: " + "; ".join(sorted(set(problems))))

    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
    REGISTRY = Registry(MAX_BOUNDARIES, [m.kv_cache_group_id for m in mamba], coord.block_pool,
                        make_block_hash_with_group_id)
    for m in mamba:
        m._glm53_rb_on = True
    for m in fa:
        m._glm53_rb_on = True
    swa[0][0]._glm53_rb_on = True
    for m in scratch:                       # explicit: a scratch group (the kpool tail) never runs a lever hunk
        m._glm53_rb_on = False
    _log().info("GLM53_REPLAY_BOUNDARY on: scheduler block %d, hash unit %d, %d KDA groups %s, MLA group %s, drafter "
                "group %d, %d scratch group(s) %s skipped, retention %s, cap %d, boot %s", scheduler.block_size,
                scheduler.hash_block_size, len(mamba), [m.kv_cache_group_id for m in mamba],
                [m.kv_cache_group_id for m in fa], swa[0][0].kv_cache_group_id, len(scratch),
                [m.kv_cache_group_id for m in scratch], getattr(coord, "retention_interval", None), MAX_BOUNDARIES,
                BOOT_ID)
    return True


__all__ = ["ENABLED", "MAX_BOUNDARIES", "BOOT_ID", "replay_boundary", "Boundary", "request_boundary", "plan_chunk",
           "Registry", "registry", "enable"]
