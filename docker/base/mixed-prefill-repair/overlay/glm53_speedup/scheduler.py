"""Scheduler-side hooks; only the existing SchedulerOutput crosses TP ranks."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

from .policy import CostTable, Policy, Settings, trim_batch
from .fair_prefill import FairPrefill

# Route opt-in scheduler records through vLLM's existing INFO handler.  The
# image root has no handler, so package INFO records otherwise disappear.
log = logging.getLogger("vllm.glm53_speedup.scheduler")
log.setLevel(logging.INFO)


def is_prefilling(request):
    # Resumed requests may be replaying previous output after eviction.
    boundary = max(request.num_prompt_tokens, getattr(request, "num_tokens", request.num_prompt_tokens) - 1)
    return request.num_computed_tokens < boundary


class SchedulerBridge:
    def __init__(self, scheduler):
        self.scope = os.environ.get("GLM53_SPEEDUP_CAPTURE", "off")
        self.override = os.environ.get("GLM53_SPEEDUP_OVERRIDE", "")
        mode = os.environ.get("GLM53_SPEEDUP_MODE", "off")
        self.policy = Policy(Settings(mode, int(os.environ.get("GLM53_SPEEDUP_K", "4")), self.scope))
        self.stamp = None
        self.last_poll = 0.0
        self.started = 0.0
        self.decision = None
        self.log_every = int(os.environ.get("GLM53_SPEEDUP_LOG_EVERY", "0"))
        self.step = 0
        self.acceptance = []
        self.scheduled_counts = {}
        self.prefill_requests = []
        self.runtime_prefill = os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME", "0") == "1"
        self.fair_prefill = (FairPrefill(scheduler.block_size)
                             if os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL_FAIR", "0") == "1"
                             else None)
        self.prefill_cap = int(os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL", "0"))
        if self.prefill_cap not in (0, 2304, 4608, 6912, 9216):
            raise ValueError("mixed prefill cap must be 0/2304/4608/6912/9216")
        if self.fair_prefill:
            self.fair_prefill.validate_cap(self.prefill_cap)
        enabled = self.scope != "off" or self.prefill_cap != 0 or self.runtime_prefill or self.fair_prefill
        if enabled:
            cfg = scheduler.vllm_config
            if (not scheduler.use_v2_model_runner or scheduler.num_spec_tokens != 4
                    or cfg.parallel_config.tensor_parallel_size != 2
                    or cfg.parallel_config.pipeline_parallel_size != 1
                    or getattr(cfg.scheduler_config, "async_scheduling", False)
                    or scheduler.num_sampled_tokens_per_step != 1):
                raise ValueError("speedup experiments require synchronous V2 TP2/PP1 DFlash2 K4")
            if (not cfg.speculative_config or cfg.speculative_config.method != "dflash"
                    or "DFlash2DraftModel" not in cfg.speculative_config.draft_model_config.architectures):
                raise ValueError("speedup experiments require DFlash2")
            if scheduler.dynamic_sd_lookup is not None:
                raise ValueError("do not combine adaptive K with the batch-size SD table")
            if self.fair_prefill:
                if getattr(scheduler.policy, "name", "") != "FCFS":
                    raise ValueError("fair prefill is qualified for FCFS only")
                if mode != "off" or int(os.environ.get("GLM53_SPEEDUP_K", "4")) != 4:
                    raise ValueError("do not combine fair prefill with adaptive K")
                if not scheduler.need_mamba_block_aligned_split:
                    raise ValueError("fair prefill requires existing hybrid block alignment")
        path = os.environ.get("GLM53_SPEEDUP_COSTS", "")
        if path:
            document = json.loads(Path(path).read_text())
            identity = os.environ.get("GLM53_SPEEDUP_COST_IDENTITY", "")
            table_identity = document.get("compatibility_identity", document.get("runtime_identity"))
            if not identity or table_identity != identity:
                raise ValueError("cost table runtime identity does not match this arm")
            self.policy.costs = CostTable.from_document(document)

    def refresh(self):
        # One scheduler reader, no per-layer/per-rank filesystem polling.
        now = time.monotonic()
        if not self.override or now - self.last_poll < 1.0:
            return
        self.last_poll = now
        try:
            path = Path(self.override)
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if stamp == self.stamp:
                return
            if stat.st_size > 4096:
                raise ValueError("override exceeds 4096 bytes")
            data = json.loads(path.read_text())
            allowed = {"mode", "forced_k"} | ({"mixed_prefill_tokens"} if self.runtime_prefill else set())
            if set(data) - allowed:
                raise ValueError("unknown override field")
            cap = data.get("mixed_prefill_tokens", 0)
            if type(cap) is not int or cap not in (0, 2304, 4608, 6912, 9216):
                raise ValueError("invalid runtime mixed prefill cap")
            setting = Settings(data.get("mode", "off"), data.get("forced_k", 4), self.scope)
            if cap and (setting.mode != "off" or setting.forced_k != 4):
                raise ValueError("do not combine mixed prefill with adaptive K")
            if self.fair_prefill:
                self.fair_prefill.validate_cap(cap)
                if setting.mode != "off" or setting.forced_k != 4:
                    raise ValueError("fair prefill requires fixed K4")
            self.policy.configure(setting)
            if self.runtime_prefill:
                self.prefill_cap = cap
            self.stamp = stamp
            log.info("GLM53_SPEEDUP_OVERRIDE %s", json.dumps(data, sort_keys=True))
        except (OSError, ValueError, TypeError) as exc:
            # A missing/truncated file cannot leave an old experimental mode on.
            self.policy.configure(Settings("off", 4, self.scope))
            if self.runtime_prefill:
                self.prefill_cap = 0
            self.stamp = None
            log.error("GLM53_SPEEDUP_OVERRIDE_INVALID %s", exc)

    def select(self, scheduler, counts, drafts, preempted):
        if self.fair_prefill:
            self.fair_prefill.observe(counts)
        if self.scope == "off" and not self.override and not self.log_every:
            return
        self.refresh()
        self.policy.retain(scheduler.requests)
        for req in preempted:
            self.policy.requests.pop(req.request_id, None)
        contexts = {rid: scheduler.requests[rid].num_computed_tokens for rid in counts}
        mixed = any(is_prefilling(scheduler.requests[rid])
                    or rid not in drafts or counts[rid] != len(drafts[rid]) + 1
                    or len(drafts[rid]) < 4 for rid in counts)
        self.decision = self.policy.choose(contexts, mixed=mixed)
        if self.decision.k < 4:
            trim_batch(counts, drafts, self.decision.k)
        self.scheduled_counts = dict(counts)
        self.prefill_requests = [rid for rid in counts if is_prefilling(scheduler.requests[rid])]
        self.started = time.monotonic()
        self.step += 1
        self.acceptance = []

    def observe(self, rid, proposed, accepted, stale=False):
        if self.scope == "off" and not self.override and not self.log_every:
            return
        if stale:
            self.policy.requests.pop(rid, None)
            return
        self.policy.observe(rid, proposed, accepted)
        self.acceptance.append({"request": rid, "proposed": proposed, "accepted": accepted})

    def finish(self):
        if self.log_every > 0 and self.decision and self.step % self.log_every == 0:
            d = self.decision
            log.info("GLM53_SPEEDUP_STEP %s", json.dumps({
                "step": self.step, "k": d.k, "reason": d.reason,
                "concurrency": d.shape.concurrency if d.shape else 0,
                "scheduled_tokens": d.shape.tokens if d.shape else None,
                "actual_scheduled_tokens": sum(self.scheduled_counts.values()),
                "scheduled_counts": self.scheduled_counts,
                "prefill_requests": self.prefill_requests,
                "mixed_prefill_cap": self.prefill_cap,
                "prefill_policy": "fair" if self.fair_prefill else "legacy",
                "prefill_grants": self.fair_prefill.grants if self.fair_prefill else None,
                "prefill_fallback": self.fair_prefill.reason if self.fair_prefill else None,
                "decode_reserved": self.fair_prefill.decode_reserved if self.fair_prefill else None,
                "kda_block_size": self.fair_prefill.block_size if self.fair_prefill else None,
                "scheduler_roundtrip_ms": (time.monotonic() - self.started) * 1000,
                "acceptance": self.acceptance,
            }, sort_keys=True))


def bridge(scheduler):
    if not hasattr(scheduler, "_glm53_speedup"):
        scheduler._glm53_speedup = SchedulerBridge(scheduler)
    return scheduler._glm53_speedup


def prefill_limit(scheduler, request, count):
    state = bridge(scheduler)
    if not state.prefill_cap or not is_prefilling(request):
        return count
    if getattr(state, "fair_prefill", None):
        return state.fair_prefill.limit(request, count)
    decoders = [r for r in scheduler.running if not is_prefilling(r)]
    if not decoders:
        return count
    return min(count, state.prefill_cap)


def order_running(scheduler):
    # Stable partition ensures that a long ongoing prefill cannot consume the
    # budget before the decoder. Existing Mamba alignment still runs afterwards.
    state = bridge(scheduler)
    if getattr(state, "runtime_prefill", False):
        state.refresh()
    if getattr(state, "fair_prefill", None):
        state.fair_prefill.plan(scheduler, state.prefill_cap, is_prefilling)
        return
    if state.prefill_cap:
        scheduler.running.sort(key=is_prefilling)


def alignment_limit(scheduler, request, count):
    # Fair mode only chooses existing block boundaries; never turn a small
    # grant into permission for the legacy private sub-block state path.
    if getattr(bridge(scheduler), "fair_prefill", None):
        return count
    return prefill_limit(scheduler, request, count)
