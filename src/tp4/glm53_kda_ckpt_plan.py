# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): a KDA checkpoint planner after SparkRing's recurrent
# prefill-checkpoint planner, with DFlash and retention intervals allowed, an unrestricted budget and block-grid
# exports, plus replay-boundary exports off the grid and from mid-block starts (GLM53_REPLAY_BOUNDARY).
"""KDA checkpoint plans for GLM-5.3-Flash prefill (kda-checkpoints lane, 2026-09-24).

A KDA (mamba "align") prefill chunk used to end at every position whose recurrent state the prefix cache keeps: the
DFlash one-block backoff (`last_cache_position`), the prompt's last hash boundary, a shared-prefix junction. With
GLM53_KDA_CKPT=1 a chunk runs past such a position when the state there can be exported from inside the chunk (a
multiple of the mamba block size, chunk-aligned from the chunk start, at most GLM53_KDA_CKPT_MAX of them): the planner
returns the new chunk end and the exported positions. The scheduler hands the plan to the mamba managers (real blocks at
the export columns instead of null padding) and to the workers (SchedulerOutput.glm53_kda_ckpt_plans); the V2 model
runner maps it to batch rows for the KDA metadata builder through `set_step` / `step` / `clear_step`.

Idea and planner shape after FujitsuPolycom/sparkring (Apache-2.0), `vllm/v1/core/recurrent_prefill_checkpoint.py` in
runtime/images/sparkring-r35/patches/vllm-sparkring.patch @ 90d510a5 (see the kda-prefill-checkpoints NOTICE).
Modified for the GLM-5.3 lab: DFlash and retention intervals allowed, budget unrestricted, block-grid exports only.
Modified by the GLM-5.3 lab (replay-boundary, 2026-09-24; outputs/2026-09-24-replay-boundary/DESIGN.md): with
GLM53_REPLAY_BOUNDARY=1 the scheduler also plans the replay boundary R(P) off the grid and plans from mid-block starts
(glm53_replay_boundary.plan_chunk); validation accepts both, and an export's column is cdiv(position, block) - 1 (the
same column as before for an on-grid position).
"""

from __future__ import annotations

import os

ENABLED = os.environ.get("GLM53_KDA_CKPT", "0") == "1"
MAX_EXPORTS = int(os.environ.get("GLM53_KDA_CKPT_MAX", "4"))
CHUNK_TOKENS = 16           # b12x KDA prefill tile: export offsets are multiples of it

Plan = tuple  # (start, end, (position, ...)) in absolute prompt tokens


def plan_chunk(start: int, end: int, stops, grid: int, max_exports: int = MAX_EXPORTS) -> tuple[int, tuple[int, ...]]:
    """Walk the chunk's mandatory stops in order: a stop on the ``grid`` (the mamba block size) that is chunk-aligned
    from ``start`` becomes an export (up to ``max_exports``); the first stop that cannot be exported ends the chunk.
    Returns (end, exports). Without exportable stops this is exactly the scheduler's own `min(stops inside)` rule."""
    if grid <= 0 or grid % CHUNK_TOKENS:
        raise ValueError(f"grid {grid} must be a positive multiple of {CHUNK_TOKENS}")
    exports: list[int] = []
    for stop in sorted({int(s) for s in stops if start < s < end}):
        if stop % grid == 0 and (stop - start) % CHUNK_TOKENS == 0 and len(exports) < max_exports:
            exports.append(stop)
            continue
        return stop, tuple(exports)
    return end, tuple(exports)


def validate_plan(plan, block_size: int) -> tuple[int, int, tuple[int, ...]]:
    start, end, exports = plan
    start, end = int(start), int(end)
    exports = tuple(int(p) for p in exports)
    if not 0 <= start < end or start % CHUNK_TOKENS:
        raise ValueError(f"KDA checkpoint plan span [{start}, {end}) must start on the {CHUNK_TOKENS}-token tile grid")
    if not exports or exports != tuple(sorted(set(exports))):
        raise ValueError("KDA checkpoint exports must be sorted, unique and non-empty")
    columns = [-(-p // block_size) - 1 for p in exports]
    for p in exports:
        if not start < p < end or (p - start) % CHUNK_TOKENS:
            raise ValueError(f"KDA checkpoint export {p} is not an interior tile state of [{start}, {end})")
    if len(set(columns)) != len(columns) or -(-end // block_size) - 1 in columns:
        raise ValueError(f"KDA checkpoint exports {exports} of [{start}, {end}) share a block column")
    return start, end, exports


def export_columns(plan, block_size: int) -> tuple[int, ...]:
    """Block-table columns that hold the exported states (column c holds the state after (c + 1) * block_size)."""
    _, _, exports = validate_plan(plan, block_size)
    return tuple(-(-p // block_size) - 1 for p in exports)


def rows_for_batch(plans: dict, req_ids, num_computed_tokens, num_scheduled_tokens) -> list:
    """Map {req_id: plan} to the worker's batch rows (a plan or None per row); the plan's span must be the row's."""
    rows: list = [None] * len(req_ids)
    index = {rid: i for i, rid in enumerate(req_ids)}
    for rid, plan in plans.items():
        i = index.get(rid)
        if i is None:
            raise ValueError(f"KDA checkpoint plan for {rid} has no row in the worker batch")
        start, end, exports = plan
        if int(num_computed_tokens[i]) != int(start) or int(num_scheduled_tokens[i]) != int(end) - int(start):
            raise ValueError(f"KDA checkpoint plan span [{start}, {end}) differs from row {i}: computed "
                             f"{int(num_computed_tokens[i])}, scheduled {int(num_scheduled_tokens[i])}")
        rows[i] = (int(start), int(end), tuple(int(p) for p in exports))
    return rows


_STEP: list | None = None


def set_step(rows: list | None) -> None:
    global _STEP
    _STEP = rows


def step() -> list | None:
    return _STEP


def clear_step() -> None:
    global _STEP
    _STEP = None


__all__ = ["ENABLED", "MAX_EXPORTS", "CHUNK_TOKENS", "plan_chunk", "validate_plan", "export_columns",
           "rows_for_batch", "set_step", "step", "clear_step"]
