"""Scheduler-side hooks; only the existing SchedulerOutput crosses TP ranks."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat as stat_module
import time

from .geometry import Geometry, capture_axes_from_env
from .policy import CostTable, Policy, Settings, trim_batch
from .adaptive_chunk import chunk_cap
from .fair_prefill import FairPrefill
from .cadence import SplitCadence

# Route opt-in scheduler records through vLLM's existing INFO handler.  The
# image root has no handler, so package INFO records otherwise disappear.
log = logging.getLogger("vllm.glm53_speedup.scheduler")
log.setLevel(logging.INFO)
MAX_COST_BYTES = 1024 * 1024


def _file_stamp(info):
    return (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)


def _read_bounded(path: Path, limit: int):
    # Limit the read itself as well as the initial size check: an in-place
    # writer must not turn a small stat result into an unbounded allocation.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat_module.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("runtime input is not a bounded regular file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    stamp = _file_stamp(before)
    if (len(raw) > limit or len(raw) != before.st_size or stamp != _file_stamp(after)
            or stamp != _file_stamp(path.stat())):
        raise ValueError("runtime input changed while reading")
    return raw, stamp


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def load_cost_table(path: str, identity: str, expected_sha256: str | None = None,
                    *, require_k4_drafter: bool = False) -> CostTable:
    """Validate one bounded byte snapshot before exposing any costs to Policy."""
    if require_k4_drafter and (
            not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("cost activation requires independent identity and table SHA256")
    raw, _ = _read_bounded(Path(path), MAX_COST_BYTES)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("cost table SHA256 differs from runtime override")
    document = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(document, dict):
        raise ValueError("cost table must be an object")
    table_identity = (document.get("compatibility_identity") if require_k4_drafter
                      else document.get("compatibility_identity", document.get("runtime_identity")))
    if not identity or table_identity != identity:
        raise ValueError("cost table runtime identity does not match runtime override")
    return CostTable.from_document(document, require_k4_drafter=require_k4_drafter)


def is_prefilling(request):
    # Resumed requests may be replaying previous output after eviction.
    boundary = max(request.num_prompt_tokens, getattr(request, "num_tokens", request.num_prompt_tokens) - 1)
    return request.num_computed_tokens < boundary


class SchedulerBridge:
    def __init__(self, scheduler):
        self.scope = os.environ.get("GLM53_SPEEDUP_CAPTURE", "off")
        self.override = os.environ.get("GLM53_SPEEDUP_OVERRIDE", "")
        self.cost_reload = os.environ.get("GLM53_SPEEDUP_COST_RELOAD", "0") == "1"
        # Only the permitted path is pinned at boot. The independent identity
        # and byte digest arrive together in the qualified atomic override.
        self.costs_path = os.environ.get("GLM53_SPEEDUP_COSTS", "")
        mode = os.environ.get("GLM53_SPEEDUP_MODE", "off")
        if self.cost_reload and mode == "cost":
            mode = "off"  # cost activation always requires a complete override
        needs_geometry = (self.scope != "off" or mode != "off"
            or os.environ.get("GLM53_SPEEDUP_GEOMETRY", "0") != "0"
            or any(os.environ.get(key, "0") != "0" for key in (
                "GLM53_SPEEDUP_MIXED_PREFILL", "GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME",
                "GLM53_SPEEDUP_MIXED_PREFILL_FAIR", "GLM53_SPEEDUP_SPLIT_CADENCE",
                "GLM53_SPEEDUP_SPLIT_FAIR")))
        # An entirely disabled bridge must not constrain the plain engine.
        self.max_k = scheduler.num_spec_tokens if needs_geometry else 4
        self.max_requests = scheduler.max_num_running_reqs if needs_geometry else 6
        self.capture_requests, self.capture_ks = capture_axes_from_env(self.max_requests, self.max_k)
        self.policy = Policy(self._settings(mode, int(os.environ.get("GLM53_SPEEDUP_K", str(self.max_k)))))
        self.stamp = None
        self.last_poll = 0.0
        self.started = 0.0
        self.decision = None
        self.log_every = int(os.environ.get("GLM53_SPEEDUP_LOG_EVERY", "0"))
        # Bounded, default-off scheduler boundary trace for correctness
        # diagnosis. It records logical counters/labels only; it never reads
        # or rewrites CUDA/KV bytes and is not a performance path.
        self.boundary_trace = os.environ.get("GLM53_SPEEDUP_BOUNDARY_TRACE", "0") == "1"
        self.boundary_trace_max = int(os.environ.get("GLM53_SPEEDUP_BOUNDARY_TRACE_MAX", "512"))
        if self.boundary_trace_max <= 0:
            raise ValueError("boundary trace max must be positive")
        self.boundary_trace_count = 0
        self._trace_scheduler = None
        self.step = 0
        self.acceptance = []
        self.scheduled_counts = {}
        self.prefill_requests = []
        composition = os.environ.get("GLM53_SPEEDUP_PREFILL_ADAPTIVE", "0")
        if composition not in ("0", "1"):
            raise ValueError("invalid prefill/adaptive selector")
        self.prefill_adaptive = composition == "1"
        split = os.environ.get("GLM53_SPEEDUP_SPLIT_CADENCE", "0")
        if split not in ("0", "1"):
            raise ValueError("invalid split cadence selector")
        self.cadence = SplitCadence() if split == "1" else None
        adaptive = os.environ.get("GLM53_SPEEDUP_SPLIT_ADAPTIVE", "0")
        if adaptive not in ("0", "1"):
            raise ValueError("invalid split/adaptive selector")
        self.split_adaptive = adaptive == "1"
        if self.split_adaptive and self.cadence is None:
            raise ValueError("split/adaptive requires split cadence")
        self.runtime_prefill = os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME", "0") == "1"
        self.fair_prefill = (FairPrefill(scheduler.block_size)
                             if os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL_FAIR", "0") == "1"
                             else None)
        fair_split = os.environ.get("GLM53_SPEEDUP_SPLIT_FAIR", "0")
        if fair_split not in ("0", "1"):
            raise ValueError("invalid split/fair selector")
        self.split_fair = fair_split == "1"
        self.prefill_cap = int(os.environ.get("GLM53_SPEEDUP_MIXED_PREFILL", "0"))
        if self.prefill_cap not in (0, 2304, 4608, 6912, 9216):
            raise ValueError("mixed prefill cap must be 0/2304/4608/6912/9216")
        if self.fair_prefill:
            self.fair_prefill.validate_cap(self.prefill_cap)
        enabled = self.scope != "off" or self.prefill_cap != 0 or self.runtime_prefill or self.fair_prefill or self.cadence is not None or self.split_fair
        if enabled:
            geometry = Geometry.runtime(self.max_requests, self.max_k)
            cfg = scheduler.vllm_config
            if (getattr(cfg.scheduler_config, "max_num_seqs", self.max_requests) != self.max_requests
                    or getattr(cfg.speculative_config, "num_speculative_tokens", self.max_k) != self.max_k):
                raise ValueError("scheduler geometry differs from configured engine geometry")
            if (not scheduler.use_v2_model_runner
                    or cfg.parallel_config.tensor_parallel_size not in (2, 4)
                    or cfg.parallel_config.pipeline_parallel_size != 1
                    or getattr(cfg.scheduler_config, "async_scheduling", False)
                    or scheduler.num_sampled_tokens_per_step != 1):
                raise ValueError("speedup experiments require synchronous V2 TP2 or TP4/PP1 DFlash2 with engine-derived geometry")
            if (not cfg.speculative_config or cfg.speculative_config.method != "dflash"
                    or "DFlash2DraftModel" not in cfg.speculative_config.draft_model_config.architectures):
                raise ValueError("speedup experiments require DFlash2")
            if scheduler.dynamic_sd_lookup is not None:
                raise ValueError("do not combine adaptive K with the batch-size SD table")
            if self.fair_prefill:
                if getattr(scheduler.policy, "name", "") != "FCFS":
                    raise ValueError("fair prefill is qualified for FCFS only")
                if not scheduler.need_mamba_block_aligned_split:
                    raise ValueError("fair prefill requires existing hybrid block alignment")
        if self.cadence is not None or self.split_fair:
            if (scheduler.vllm_config.parallel_config.tensor_parallel_size not in (2, 4)
                    or getattr(scheduler.vllm_config.parallel_config, "data_parallel_size", 1) != 1
                    or getattr(scheduler.policy, "name", "") != "FCFS"
                    or not scheduler.need_mamba_block_aligned_split
                    or scheduler.block_size != {2: 4608, 4: 2304}.get(scheduler.vllm_config.parallel_config.tensor_parallel_size)
                    or scheduler.scheduler_config.max_num_batched_tokens != 14336
                    or scheduler.connector is not None
                    or scheduler.ec_connector is not None
                    or scheduler.is_encoder_decoder or scheduler.lora_config is not None):
                raise ValueError("split cadence requires synchronous TP2/TP4 FCFS, topology-specific block alignment and budget14336 without connectors")
        self.fair_block_size = scheduler.block_size
        self.fair_cap_geometry = os.environ.get("GLM53_SPEEDUP_TP2_FAIR_CAP_GEOMETRY", "0")
        if self.fair_cap_geometry not in ("0", "1"):
            raise ValueError("invalid TP2 fair-cap selector")
        if self.fair_cap_geometry == "1" and (
            scheduler.vllm_config.parallel_config.tensor_parallel_size != 2
            or self.fair_block_size != 4608 or not self.split_fair or not self.fair_prefill):
            raise ValueError("extended TP2 fair caps require TP2 split/fair and block4608")
        self.allowed_fair_caps = (4608, 9216) if self.fair_cap_geometry == "1" else (4608,)
        self._validate_prefill_adaptive(self.policy.settings, self.prefill_cap)
        adaptive_chunk = os.environ.get("GLM53_ADAPTIVE_FAIR_CHUNK", "0")
        if adaptive_chunk not in ("0", "1"):
            raise ValueError("invalid adaptive fair chunk selector")
        self.adaptive_chunk = adaptive_chunk == "1"
        self.effective_prefill_cap = self.prefill_cap
        if self.adaptive_chunk and (not self.split_fair or not self.fair_prefill
                or self.runtime_prefill or self.prefill_cap not in self.allowed_fair_caps):
            raise ValueError("adaptive chunk requires static aligned split/fair caps")
        if self.adaptive_chunk:
            low = chunk_cap(self.prefill_cap, 5, True, max_requests=self.max_requests,
                            block_size=self.fair_block_size)
            self.fair_prefill.validate_cap(low)
            log.info("GLM53_ADAPTIVE_FAIR_CHUNK selected=True decoder_threshold=5 low=%d high=%d block=%d varying=%s",
                     low, self.prefill_cap, self.fair_block_size, low != self.prefill_cap)
        path = self.costs_path
        if path and not self.cost_reload:
            # Preserve the legacy static-table contract when reload is off.
            document = json.loads(Path(path).read_text())
            identity = os.environ.get("GLM53_SPEEDUP_COST_IDENTITY", "")
            table_identity = document.get("compatibility_identity", document.get("runtime_identity"))
            if not identity or table_identity != identity:
                raise ValueError("cost table runtime identity does not match this arm")
            self.policy.costs = CostTable.from_document(document)

        if enabled:
            log.info("GLM53_SPEEDUP_GEOMETRY selected=True consumer=scheduler requests=%d proposal_k=%d", geometry.max_requests, geometry.max_k)

        if self.policy.settings.mode in ("ema4", "saturation"):
            log.info("GLM53_SPEEDUP_SATURATION selected=%s mode=%s native_k=%d baseline_k=4 required_full_accepts=8",
                     self.policy.settings.mode == "saturation", self.policy.settings.mode, self.max_k)

    def _settings(self, mode, k):
        return Settings(mode, k, self.scope, self.max_requests, self.max_k, self.capture_requests, self.capture_ks)

    def _validate_prefill_adaptive(self, setting, cap):
        if self.split_fair:
            if (setting.mode not in ("off", "ema", "forced", "ema4", "saturation") or (setting.mode != "forced" and setting.forced_k != self.max_k)
                    or self.scope != "all" or cap not in self.allowed_fair_caps or not self.fair_prefill
                    or self.split_adaptive or not self.prefill_adaptive
                    or self.runtime_prefill or self.cost_reload or self.costs_path):
                raise ValueError("split/fair requires all capture, static aligned fair cap and off/EMA/forced/ema4/saturation only")
        elif self.cadence is not None and self.split_adaptive:
            if (setting.mode not in ("off", "ema", "forced", "ema4", "saturation") or (setting.mode != "forced" and setting.forced_k != self.max_k)
                    or self.scope != "all" or cap != 4608 or self.fair_prefill
                    or not self.prefill_adaptive or self.runtime_prefill
                    or self.cost_reload or self.costs_path):
                raise ValueError("split/adaptive requires all capture, static legacy cap4608 and off/EMA/forced/ema4/saturation only")
        elif self.cadence is not None and (
                setting.mode != "off" or setting.forced_k != self.max_k
                or self.scope != "off" or cap != 4608 or self.fair_prefill
                or self.prefill_adaptive or self.runtime_prefill
                or self.cost_reload or self.costs_path):
            raise ValueError("split cadence requires static legacy cap4608 and fixed K4; no adaptive or cost composition")
        if not (cap or self.fair_prefill) or (setting.mode == "off" and setting.forced_k == self.max_k):
            return
        # Admission still reserves the complete configured-K query and draft scratch.
        # Policy.choose keeps every mixed/short/resumed batch at configured K; only a
        # complete pure decode batch trims, after all allocation decisions.
        if (not self.prefill_adaptive or setting.mode not in ("ema", "forced", "ema4", "saturation")
                or self.scope != "all" or cap not in self.allowed_fair_caps or self.runtime_prefill
                or self.cost_reload or self.costs_path
                or (setting.mode in ("ema", "ema4", "saturation") and setting.forced_k != self.max_k)):
            raise ValueError("prefill/adaptive composition requires explicit opt-in, static aligned cap, all capture and EMA/forced/ema4/saturation only")

    @staticmethod
    def _request_boundary(request):
        """Return bounded scheduler/request state at one logical boundary."""
        status = getattr(getattr(request, "status", None), "name", None)
        fields = (
            "request_id", "num_prompt_tokens", "num_tokens", "num_computed_tokens",
            "num_tokens_with_spec", "num_output_placeholders", "num_in_flight_tokens",
            "num_stale_output_tokens", "num_preemptions", "is_prefill_chunk",
            "shared_prefix_boundary", "num_output_tokens",
        )
        result = {name: getattr(request, name, None) for name in fields}
        result["status"] = status
        return result

    def _trace_boundary(self, stage, scheduler, counts, prefill_requests):
        """Emit one compact JSON boundary record for a dedicated diagnostic arm."""
        if not self.boundary_trace or self.boundary_trace_count >= self.boundary_trace_max:
            return
        rows = []
        for rid in sorted(counts):
            request = scheduler.requests.get(rid)
            if request is not None:
                rows.append(self._request_boundary(request))
        payload = {
            "stage": stage,
            "step": self.step,
            "mixed_prefill_cap": self.prefill_cap,
                "effective_prefill_cap": self.effective_prefill_cap,
            "scheduled_counts": dict(counts),
            "prefill_requests": list(prefill_requests),
            "actual_scheduled_tokens": sum(counts.values()),
            "requests": rows,
        }
        self.boundary_trace_count += 1
        log.info("GLM53_SPEEDUP_BOUNDARY %s", json.dumps(payload, sort_keys=True))

    def refresh(self):
        # One scheduler reader, no per-layer/per-rank filesystem polling.
        now = time.monotonic()
        if now - self.last_poll < 1.0:
            return
        if not self.override and not self.cost_reload:
            return
        self.last_poll = now
        try:
            if not self.override:
                raise ValueError("cost reload requires an override path")
            path = Path(self.override)
            stat = path.stat()
            stamp = _file_stamp(stat)
            if stamp == self.stamp and not self.cost_reload:
                return
            raw, stamp = _read_bounded(path, 4096)
            data = json.loads(raw, object_pairs_hook=_unique_object)
            if not isinstance(data, dict):
                raise ValueError("override must be an object")
            cost_fields = {"costs_path", "cost_identity", "cost_sha256"}
            allowed = {"mode", "forced_k"} | ({"mixed_prefill_tokens"} if self.runtime_prefill else set())
            if self.cost_reload:
                allowed |= cost_fields
            if set(data) - allowed:
                raise ValueError("unknown override field")
            cap = data.get("mixed_prefill_tokens", 0)
            if type(cap) is not int or cap not in (0, 2304, 4608, 6912, 9216):
                raise ValueError("invalid runtime mixed prefill cap")
            setting = self._settings(data.get("mode", "off"), data.get("forced_k", self.max_k))
            effective_cap = cap if self.runtime_prefill else self.prefill_cap
            self._validate_prefill_adaptive(setting, effective_cap)
            if self.fair_prefill:
                self.fair_prefill.validate_cap(effective_cap)
            costs = self.policy.costs
            if self.cost_reload:
                costs = None
                if setting.mode == "cost":
                    if set(data) != {"mode"} | cost_fields:
                        raise ValueError("cost override requires exactly path, identity and SHA256")
                    selected = data["costs_path"]
                    if (not isinstance(selected, str) or not selected or selected != self.costs_path
                            or not Path(selected).is_absolute()
                            or str(Path(selected).resolve()) != selected):
                        raise ValueError("cost path must equal the canonical permitted startup path")
                    cost_stamp = _file_stamp(Path(selected).stat())
                    costs = load_cost_table(selected, data["cost_identity"], data["cost_sha256"],
                                            require_k4_drafter=True)
                    if (str(Path(selected).resolve()) != selected
                            or _file_stamp(Path(selected).stat()) != cost_stamp):
                        raise ValueError("cost path changed while reading")
                elif set(data) & cost_fields:
                    raise ValueError("cost fields are only permitted in cost mode")
            # Active reloads validate the digest every second even when the
            # override inode is unchanged. Qualify control/arm overhead alike.
            if _file_stamp(path.stat()) != stamp:
                raise ValueError("override changed during validation")
            changed = stamp != self.stamp or setting != self.policy.settings or costs != self.policy.costs
            # The cost-reload API is the only path that needs the newer
            # keyword.  A static/off run must remain callable with the legacy
            # Policy.configure(settings) implementation that is present in
            # older candidate images.  The paired source bundle still pins
            # both implementations for reload-enabled arms.
            if self.cost_reload:
                self.policy.configure(setting, costs=costs)
            else:
                self.policy.configure(setting)
            if self.runtime_prefill:
                self.prefill_cap = cap
            self.stamp = stamp
            if changed:
                log.info("GLM53_SPEEDUP_OVERRIDE %s", json.dumps(data, sort_keys=True))
        except (OSError, ValueError, TypeError, OverflowError, RuntimeError) as exc:
            # A missing/truncated file cannot leave an old experimental mode on.
            if self.cost_reload:
                self.policy.configure(self._settings("off", self.max_k), costs=None)
            else:
                self.policy.configure(self._settings("off", self.max_k))
            if self.runtime_prefill:
                self.prefill_cap = 0
            self.stamp = None
            log.error("GLM53_SPEEDUP_OVERRIDE_INVALID %s", exc)

    def select(self, scheduler, counts, drafts, preempted):
        if self.cadence is not None:
            self.cadence.observe(counts)
        # No prefill offer was made during the owed decode-only turn.
        # Preserve fair queue/debt exactly instead of spending priority
        # for a grant that cadence deliberately withheld.
        if self.fair_prefill and not (self.split_fair and self.cadence is not None
                                      and self.cadence.phase == "decode"):
            self.fair_prefill.observe(counts)
        if self.scope == "off" and not self.override and not self.log_every:
            return
        self.refresh()
        self.policy.retain(scheduler.requests)
        for req in preempted:
            self.policy.discard(req.request_id)
        contexts = {rid: scheduler.requests[rid].num_computed_tokens for rid in counts}
        mixed = any(is_prefilling(scheduler.requests[rid])
                    or rid not in drafts or counts[rid] != len(drafts[rid]) + 1
                    or len(drafts[rid]) != self.max_k for rid in counts)
        self.decision = self.policy.choose(contexts, mixed=mixed)
        if not mixed and self.decision.k < self.max_k:
            trim_batch(counts, drafts, self.decision.k, max_k=self.max_k)
        self.scheduled_counts = dict(counts)
        self.prefill_requests = [rid for rid in counts if is_prefilling(scheduler.requests[rid])]
        self.started = time.monotonic()
        self.step += 1
        self.acceptance = []
        self._trace_scheduler = scheduler
        self._trace_boundary("select", scheduler, counts, self.prefill_requests)

    def observe(self, rid, proposed, accepted, stale=False):
        if self.scope == "off" and not self.override and not self.log_every:
            return
        if stale:
            self.policy.discard(rid)
            return
        self.policy.observe(rid, proposed, accepted)
        self.acceptance.append({"request": rid, "proposed": proposed, "accepted": accepted})

    def finish(self):
        if self._trace_scheduler is not None:
            self._trace_boundary("finish", self._trace_scheduler,
                                 self.scheduled_counts, self.prefill_requests)
        if self.log_every > 0 and self.decision and self.step % self.log_every == 0:
            d = self.decision
            log.info("GLM53_SPEEDUP_STEP %s", json.dumps({
                "step": self.step, "k": d.k, "proposal_k": self.max_k, "max_requests": self.max_requests, "reason": d.reason,
                "concurrency": d.shape.concurrency if d.shape else 0,
                "scheduled_tokens": d.shape.tokens if d.shape else None,
                "actual_scheduled_tokens": sum(self.scheduled_counts.values()),
                "scheduled_counts": self.scheduled_counts,
                "prefill_requests": self.prefill_requests,
                "mixed_prefill_cap": self.prefill_cap,
                "effective_prefill_cap": self.effective_prefill_cap,
                "prefill_policy": "fair" if self.fair_prefill else "legacy",
                "split_cadence": self.cadence.snapshot() if self.cadence else None,
                "prefill_grants": self.fair_prefill.grants if self.fair_prefill else None,
                "prefill_fallback": self.fair_prefill.reason if self.fair_prefill else None,
                "decode_reserved": self.fair_prefill.decode_reserved if self.fair_prefill else None,
                "kda_block_size": self.fair_prefill.block_size if self.fair_prefill else None,
                "scheduler_roundtrip_ms": (time.monotonic() - self.started) * 1000,
                "acceptance": self.acceptance,
                "saturation": self.policy.snapshot(),
            }, sort_keys=True))
        self._trace_scheduler = None


def bridge(scheduler):
    if not hasattr(scheduler, "_glm53_speedup"):
        scheduler._glm53_speedup = SchedulerBridge(scheduler)
    return scheduler._glm53_speedup


def prefill_limit(scheduler, request, count):
    state = bridge(scheduler)
    if state.cadence is not None:
        count = state.cadence.limit(request, count)
    if not state.prefill_cap or not is_prefilling(request):
        return count
    if getattr(state, "fair_prefill", None):
        return state.fair_prefill.limit(request, count)
    decoders = [r for r in scheduler.running if not is_prefilling(r)]
    if not decoders:
        return count
    return min(count, state.prefill_cap)


def defer_waiting(scheduler, request):
    """A fair-policy zero grant defers this waiter, not the whole queue."""
    state = bridge(scheduler)
    if state.cadence is not None and state.cadence.deferred(request):
        return True
    fair = getattr(state, "fair_prefill", None)
    return bool(state.prefill_cap and fair is not None
                and fair.grants is not None
                and fair.grants.get(request.request_id) == 0)


def order_running(scheduler):
    # Stable partition ensures that a long ongoing prefill cannot consume the
    # budget before the decoder. Existing Mamba alignment still runs afterwards.
    state = bridge(scheduler)
    if getattr(state, "runtime_prefill", False):
        state.refresh()
    if getattr(state, "fair_prefill", None):
        decoder_count = sum(not is_prefilling(r) for r in scheduler.running)
        effective_cap = chunk_cap(state.prefill_cap, decoder_count, state.adaptive_chunk, max_requests=state.max_requests, block_size=state.fair_block_size)
        state.effective_prefill_cap = effective_cap
        if state.split_fair and state.cadence is not None:
            state.cadence.begin(scheduler, state.effective_prefill_cap, is_prefilling)
            if state.cadence.phase == "decode":
                state.fair_prefill.defer_for_decode(
                    len(state.cadence.decoder_ids) * (scheduler.num_spec_tokens + 1))
                return
        # Retain the actual running list, including all decoders. Fair
        # grants therefore keep their cap and conservative full configured-K
        # input/scratch reservations in a prefill-only turn.
        state.fair_prefill.plan(scheduler, effective_cap, is_prefilling)
        return
    if state.prefill_cap:
        scheduler.running.sort(key=is_prefilling)
    if state.cadence is not None:
        state.cadence.begin(scheduler, state.prefill_cap, is_prefilling)


def alignment_limit(scheduler, request, count):
    # Fair mode only chooses existing block boundaries; never turn a small
    # grant into permission for the legacy private sub-block state path.
    if getattr(bridge(scheduler), "fair_prefill", None):
        return count
    return prefill_limit(scheduler, request, count)
