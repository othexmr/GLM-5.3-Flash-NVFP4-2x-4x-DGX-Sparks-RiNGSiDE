"""CPU-only, reproducible verification-length policy; no vLLM imports."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
import statistics
from typing import Mapping
from .geometry import Geometry, capture_axes

KS = (1, 2, 3, 4)
CONTEXTS = (8192, 32768, 131072)
# Cost captures at a nominal context include the fixed 256-token generation
# span.  This is a bounded coverage allowance, not extrapolation of an 8K curve
# to arbitrary longer contexts.
DEFAULT_OUTPUT_HEADROOM = 256
_KEEP_COSTS = object()


def reject_test_costs(value):
    """Reject explicit CPU/fixture labels, including labels nested in rows."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).lower().replace("-", "_")
            if name in ("synthetic", "fixture", "test_only", "capture_only", "cpu_only") and item:
                raise ValueError("synthetic or test-only cost tables cannot serve")
            if name in ("measurement_origin", "capture", "status", "origin") and isinstance(item, str):
                marker = item.lower().replace("-", "_")
                if (marker.startswith(("cpu_", "synthetic", "fixture"))
                        or "test_only" in marker or "capture_only" in marker):
                    raise ValueError("synthetic or test-only cost tables cannot serve")
            reject_test_costs(item)
    elif isinstance(value, list):
        for item in value:
            reject_test_costs(item)


@dataclass(frozen=True, order=True)
class DecodeShape:
    concurrency: int
    k: int

    @property
    def query_len(self):
        return self.k + 1

    @property
    def tokens(self):
        return self.concurrency * self.query_len


def decode_shapes(scope: str, max_requests: int = 6, max_k: int = 4, *, requests=None, ks=None) -> tuple[DecodeShape, ...]:
    if scope not in ("off", "c1c2", "all"):
        raise ValueError("capture scope must be off, c1c2, or all")
    requests, ks = capture_axes(max_requests, max_k, requests, ks)
    return tuple(DecodeShape(c, k) for c in range(1, max_requests + 1)
                 for k in (ks if c in requests and (scope == "all" or scope == "c1c2" and c <= 2) else (max_k,)))


@dataclass(frozen=True)
class Settings:
    mode: str = "off"
    forced_k: int = 4
    scope: str = "off"
    max_requests: int = 6
    max_k: int = 4
    capture_requests: tuple[int, ...] | None = None
    capture_ks: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.mode not in ("off", "forced", "ema", "cost", "ema4", "saturation"):
            raise ValueError("mode must be off, forced, ema, cost, ema4, or saturation")
        if type(self.forced_k) is not int or self.forced_k not in Geometry(self.max_requests, self.max_k).ks:
            raise ValueError("forced_k must be an integer within configured proposal K")
        decode_shapes(self.scope, self.max_requests, self.max_k, requests=self.capture_requests, ks=self.capture_ks)
        _, ks = capture_axes(self.max_requests, self.max_k, self.capture_requests, self.capture_ks)
        if self.mode == "forced" and self.forced_k not in ks:
            raise ValueError("forced K is outside captured K set")
        if self.mode in ("ema4", "saturation"):
            cs, _ = capture_axes(self.max_requests, self.max_k, self.capture_requests, self.capture_ks)
            if (self.max_k != 7 or self.scope != "all" or self.forced_k != 7
                    or cs != tuple(range(1, self.max_requests + 1))
                    or ks != tuple(range(1, 8))):
                raise ValueError("saturation comparison requires native K7 and complete C x K1..7 capture")
        if self.mode == "cost" and (self.max_k != 4 or self.max_requests > 6 or ks != KS):
            raise ValueError("cost tables remain qualified only for C<=6/K4")
        if self.mode != "off" and self.scope == "off":
            raise ValueError("adaptive modes require captured experimental shapes")


@dataclass
class RequestHistory:
    steps: int = 0
    accepted_ema: float | None = None
    survival: list[float] | None = None
    observations: list[int] | None = None

    max_k: int = 4

    def __post_init__(self):
        Geometry(1, self.max_k)
        if self.accepted_ema is None:self.accepted_ema = float(self.max_k)
        if self.survival is None:self.survival = [1.0] * self.max_k
        if self.observations is None:self.observations = [0] * self.max_k
        if len(self.survival) != self.max_k or len(self.observations) != self.max_k:
            raise ValueError("history arrays must match proposal K")

    def observe(self, proposed: int, accepted: int):
        if type(proposed) is not int or type(accepted) is not int:
            raise ValueError("acceptance counts must be integers")
        if not 1 <= proposed <= self.max_k or not 0 <= accepted <= proposed:
            raise ValueError("invalid proposed/accepted prefix")
        # The length estimator only sees full configured-K draws; a smaller K right-censors
        # accepted length. Updating it as a full-length failure would lock the policy low.
        if proposed == self.max_k:
            self.accepted_ema = .8 * self.accepted_ema + .2 * accepted
        for j in range(proposed):
            self.survival[j] = .8 * self.survival[j] + .2 * float(accepted > j)
            self.observations[j] += 1
        self.steps += 1

    def expected_tokens(self, k: int) -> float:
        # Independent EMA updates can momentarily violate monotonicity after
        # censored samples. A survival probability cannot increase with position.
        running = 1.0
        total = 1.0  # target-sampled bonus token
        for p in self.survival[:k]:
            running = min(running, p)
            total += running
        return total


def expected_yield(prefix_lengths, k: int) -> float:
    """Return ``1 + sum(P(prefix_length >= j), j=1..K)``.

    Inputs must be verified prefixes from full K4 proposals. This helper has
    no proposal-length argument and cannot interpret censored shorter draws;
    use RequestHistory.observe(proposed, accepted) for those observations.
    """
    if type(k) is not int or k not in KS:
        raise ValueError("K outside qualified range")
    if not isinstance(prefix_lengths, (list, tuple)) or not prefix_lengths:
        raise ValueError("prefix lengths are required")
    lengths = []
    for value in prefix_lengths:
        if type(value) is not int or not 0 <= value <= 4:
            raise ValueError("invalid observed prefix length")
        lengths.append(value)
    return 1.0 + sum(sum(length >= j for length in lengths) / len(lengths)
                     for j in range(1, k + 1))


@dataclass(frozen=True)
class CostTable:
    """Measured whole-step ms, including fixed K4 drafting and common overhead."""
    costs: Mapping[tuple[int, int, int], float]
    execution_kind: str = "unspecified_legacy"
    output_headroom: int = DEFAULT_OUTPUT_HEADROOM
    compatibility_identity: str | None = None

    @classmethod
    def from_document(cls, document: dict, *, require_k4_drafter: bool = False):
        if not isinstance(document, Mapping):
            raise ValueError("cost table must be an object")
        reject_test_costs(document)
        if document.get("schema") != "glm53.speedup.costs.v1":
            raise ValueError("unsupported cost table schema")
        if document.get("measurement") != "whole_step_ms":
            raise ValueError("costs must include drafting, verification and overhead")
        runtime_identity = document.get("compatibility_identity", document.get("runtime_identity"))
        if not runtime_identity or not document.get("source_receipts"):
            raise ValueError("cost table needs runtime identity and source receipts")
        if not isinstance(runtime_identity, str) or not runtime_identity:
            raise ValueError("cost table runtime identity must be a string")
        if document.get("compatibility_identity") is not None and document.get("runtime_identity") \
                and document["compatibility_identity"] != document["runtime_identity"] \
                and document.get("historical_runtime_identity") is None:
            # A table may retain a legacy runtime ID alongside a new binding,
            # but it must say so explicitly rather than silently relabeling it.
            raise ValueError("independent and historical cost identities need an explicit relation")
        execution_kind = document.get("execution_kind", "unspecified_legacy")
        if execution_kind not in ("unspecified_legacy", "k4_drafter_trimmed", "native_k"):
            raise ValueError("unknown cost execution kind")
        if require_k4_drafter:
            if execution_kind != "k4_drafter_trimmed":
                raise ValueError("cost activation requires K4 drafter-trimmed rows")
            if (document.get("measurement_origin") != "gpu_runtime"
                    or document.get("compatibility_contract") != "glm53.cost.compatibility.v1"
                    or not isinstance(document.get("compatibility_identity"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", document["compatibility_identity"])):
                raise ValueError("cost activation needs measured GPU rows and independent binding")
            receipts = document["source_receipts"]
            if (not isinstance(receipts, dict) or any(
                    not isinstance(path, str) or not path or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for path, digest in receipts.items())):
                raise ValueError("cost activation needs source receipt SHA256 hashes")
        output_headroom = document.get("output_headroom_tokens", DEFAULT_OUTPUT_HEADROOM)
        if type(output_headroom) is not int or output_headroom < 0 or output_headroom > 2048:
            raise ValueError("invalid cost output headroom")
        if not isinstance(document.get("rows"), list) or not document["rows"]:
            raise ValueError("cost table has no rows")
        costs = {}
        for row in document["rows"]:
            if not isinstance(row, Mapping):
                raise ValueError("invalid or unqualified cost row")
            c, k, context = row.get("concurrency"), row.get("k"), row.get("context_tokens")
            ms = row.get("median_ms")
            samples = row.get("samples")
            if (type(c) is not int or c not in range(1, 7) or type(k) is not int
                    or k not in KS or type(context) is not int or context not in CONTEXTS
                    or isinstance(ms, bool) or not isinstance(ms, (int, float))
                    or not math.isfinite(ms) or ms <= 0 or type(samples) is not int
                    or samples < 5 or row.get("graph") != "FULL"):
                raise ValueError("invalid or unqualified cost row")
            key = (c, context, k)
            if key in costs:
                raise ValueError("duplicate cost row")
            if execution_kind == "native_k" and row.get("native_k") is not None \
                    and row["native_k"] != k:
                raise ValueError("native-K row does not match its K")
            if require_k4_drafter:
                if (row.get("execution_kind", execution_kind) != execution_kind
                        or "native_k" in row
                        or type(row.get("proposal_k", 4)) is not int or row.get("proposal_k", 4) != 4
                        or type(row.get("target_trim_k", k)) is not int or row.get("target_trim_k", k) != k):
                    raise ValueError("cost row is not a K4 drafter-trimmed measurement")
                if "whole_step_ms" in row:
                    timings = row["whole_step_ms"]
                    if (not isinstance(timings, list) or len(timings) != samples
                            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                   or not math.isfinite(x) or x <= 0 for x in timings)
                            or not math.isclose(statistics.median(timings), ms, rel_tol=1e-9)):
                        raise ValueError("cost row whole-step samples differ from its summary")
            costs[key] = float(ms)
        return cls(costs, execution_kind, output_headroom,
                   document.get("compatibility_identity", document.get("runtime_identity")))

    def curve(self, c: int, context: int, *, output_headroom: int = 0,
              include_measured_output_span: bool = False):
        """Return a complete measured curve covering the computed length.

        A nominal 8K row remains usable through its recorded output headroom
        (the fixed 256-token span in the saved captures).  Once that bound is
        crossed, the next measured context is required; an 8K-only table never
        becomes an invented 32K/131K curve.
        """
        if type(c) is not int or c not in range(1, 7):
            return None
        if type(context) is not int or context < 0:
            return None
        if type(output_headroom) is not int or output_headroom < 0:
            return None
        required = context + output_headroom
        bucket = None
        measured_contexts = {key[1] for key in self.costs}
        for candidate in CONTEXTS:
            if candidate in measured_contexts:
                # A caller must opt into the measured output span.  A bare
                # lookup at 8193 therefore cannot silently reuse an 8K row.
                allowance = self.output_headroom if (output_headroom or include_measured_output_span) else 0
                if required <= candidate + allowance:
                    bucket = candidate
                    break
        if bucket is None or any((c, bucket, k) not in self.costs for k in KS):
            return None
        return {k: self.costs[c, bucket, k] for k in KS}

    def eligible_concurrencies(self, context: int = 8192) -> tuple[int, ...]:
        """Complete measured curves at a context, for scoped policy reporting."""
        return tuple(c for c in range(1, 7) if self.curve(c, context) is not None)


@dataclass(frozen=True)
class Decision:
    k: int
    reason: str
    shape: DecodeShape | None


class SaturationGrowth:
    """Uniform batch growth from verified native prefixes; never SSE counts.

    Each member owes one observation per chosen turn. Missing, duplicate or
    censored observations break its streak. Membership changes and fallback
    discard all evidence. A failed larger-K probe returns to K4 immediately.
    """
    REQUIRED = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self.cap = 4
        self.members = ()
        self.streaks = {}
        self.pending = set()
        self.active = False
        self.rejected = False

    def choose(self, members, baseline_k):
        rejected = self.rejected
        if members != self.members or baseline_k < 4:
            self.reset()
            self.members = members
            self.streaks = dict.fromkeys(members, 0)
        # At a larger cap, missing feedback withdraws permission to spend it.
        if self.pending and self.cap > 4:
            self.reset()
            self.members = members
            self.streaks = dict.fromkeys(members, 0)
            rejected = True
        # Unobserved members cannot carry consecutive-success evidence forward.
        for rid in self.pending:
            self.streaks[rid] = 0
        self.pending.clear()
        self.rejected = False
        if baseline_k < 4:
            return baseline_k, "saturation_ema4"
        reason = "saturation_fallback" if rejected else "saturation_hold"
        if self.cap < 7 and members and all(self.streaks.get(r, 0) >= self.REQUIRED for r in members):
            self.cap += 1
            self.streaks = dict.fromkeys(members, 0)
            reason = "saturation_grow"
        self.active = True
        self.pending = set(members)
        return self.cap, reason

    def observe(self, rid, proposed, accepted):
        if not self.active or rid not in self.members:
            return
        if rid not in self.pending:
            # Duplicate callback is not a second engine step.
            self.streaks[rid] = 0
            if self.cap > 4:
                members = self.members
                self.reset()
                self.members = members
                self.streaks = dict.fromkeys(members, 0)
                self.rejected = True
            return
        self.pending.remove(rid)
        if proposed != self.cap or accepted != proposed:
            self.streaks[rid] = 0
            if self.cap > 4:
                members = self.members
                self.reset()
                self.members = members
                self.streaks = dict.fromkeys(members, 0)
                self.rejected = True
            return
        self.streaks[rid] = min(self.REQUIRED, self.streaks.get(rid, 0) + 1)

    def snapshot(self):
        return {"cap": self.cap, "required_full_accepts": self.REQUIRED,
                "streaks": dict(self.streaks), "awaiting": sorted(self.pending),
                "active": self.active, "rejected": self.rejected}


class Policy:
    def __init__(self, settings=Settings(), costs: CostTable | None = None):
        self.settings = settings
        self.costs = costs
        self.requests: dict[str, RequestHistory] = {}
        self.current_k = settings.max_k
        self.pending_k = settings.max_k
        self.pending_count = 0
        self.steps = 0
        self.members: tuple[str, ...] = ()
        self.baseline = None
        self.growth = None
        self._reset_extension()

    def _reset_extension(self):
        self.baseline = None
        self.growth = None
        if self.settings.mode in ("ema4", "saturation"):
            # A verified K>=4 draw contains a verified K4 prefix. Shorter
            # draws remain censored; they do not invent a failure at position 4.
            self.baseline = Policy(Settings("ema", 4, "all", self.settings.max_requests, 4))
            self.growth = SaturationGrowth()

    def discard(self, request_id):
        self.requests.pop(request_id, None)
        if self.baseline is not None:
            self.baseline.discard(request_id)
            self.growth.reset()

    def snapshot(self):
        return self.growth.snapshot() if self.growth is not None else None

    def configure(self, settings: Settings, *, costs=_KEEP_COSTS):
        # SchedulerBridge stages and validates the entire override before this
        # single scheduler-thread commit. A changed curve also resets hysteresis.
        if (settings.max_k, settings.max_requests, settings.capture_requests, settings.capture_ks) != (self.settings.max_k, self.settings.max_requests, self.settings.capture_requests, self.settings.capture_ks):
            raise ValueError("runtime geometry is immutable; requires restart and recapture")
        next_costs = self.costs if costs is _KEEP_COSTS else costs
        if settings != self.settings or next_costs != self.costs:
            self.settings, self.costs = settings, next_costs
            self.current_k = self.pending_k = settings.max_k
            self.pending_count = 0
            self._reset_extension()

    def retain(self, request_ids):
        live = set(request_ids)
        self.requests = {r: h for r, h in self.requests.items() if r in live}
        if self.baseline is not None:
            self.baseline.retain(live)
            if not set(self.growth.members) <= live:
                self.growth.reset()

    def observe(self, request_id, proposed, accepted):
        self.requests.setdefault(request_id, RequestHistory(max_k=self.settings.max_k)).observe(proposed, accepted)
        if self.baseline is not None:
            self.baseline.observe(request_id, min(proposed, 4), min(accepted, 4))
            if self.settings.mode == "saturation":
                self.growth.observe(request_id, proposed, accepted)

    def choose(self, request_contexts: Mapping[str, int], *, mixed=False) -> Decision:
        c = len(request_contexts)
        def result(k, reason):
            return Decision(k, reason, DecodeShape(c, k) if 1 <= c <= self.settings.max_requests else None)
        if self.baseline is not None:
            if not 1 <= c <= self.settings.max_requests or mixed:
                self.growth.reset()
                return result(self.settings.max_k, "mixed_or_unsupported_batch")
            base = self.baseline.choose(request_contexts)
            if self.settings.mode == "ema4":
                return result(base.k, "ema4_" + base.reason)
            k, reason = self.growth.choose(tuple(sorted(request_contexts)), base.k)
            return result(k, reason)
        if self.settings.mode == "off":
            return result(self.settings.max_k, "disabled")
        if not 1 <= c <= self.settings.max_requests or mixed:
            return result(self.settings.max_k, "mixed_or_unsupported_batch")
        captured_c, captured_k = capture_axes(self.settings.max_requests, self.settings.max_k, self.settings.capture_requests, self.settings.capture_ks)
        if c not in captured_c or (self.settings.scope == "c1c2" and c > 2):
            return result(self.settings.max_k, "outside_capture_scope")
        members = tuple(sorted(request_contexts))
        if members != self.members:
            self.members = members
            self.current_k = self.pending_k = self.settings.max_k
            self.pending_count = 0
        if self.settings.mode == "forced":
            return result(self.settings.forced_k, "forced")
        histories = [self.requests.setdefault(r, RequestHistory(max_k=self.settings.max_k)) for r in members]
        self.steps += 1
        if any(h.steps < 8 for h in histories):
            return result(self.settings.max_k, "warmup")
        if self.steps % 16 == 0:
            return result(self.settings.max_k, "full_length_probe")
        if self.settings.mode == "ema":
            desired = max(max(1, min(self.settings.max_k, math.ceil(h.accepted_ema + 1))) for h in histories)
        else:
            if self.costs and self.costs.execution_kind == "native_k":
                return result(self.settings.max_k, "cost_execution_kind_mismatch")
            curve = (self.costs.curve(c, max(request_contexts.values()),
                                      include_measured_output_span=True)
                     if self.costs else None)
            if curve is None:
                return result(self.settings.max_k, "unqualified_cost_curve")
            desired = max(KS, key=lambda k: (sum(h.expected_tokens(k) for h in histories) / curve[k], k))
        # Round upward into the explicitly captured set; no uncaptured trim.
        desired = next(k for k in captured_k if k >= desired)
        if desired == self.current_k:
            self.pending_count = 0
        else:
            self.pending_count = self.pending_count + 1 if desired == self.pending_k else 1
            self.pending_k = desired
            if self.pending_count >= 4:
                self.current_k, self.pending_count = desired, 0
        return result(self.current_k, self.settings.mode)


def trim_batch(counts: dict[str, int], drafts: dict[str, list[int]], k: int, *, max_k: int = 4) -> int:
    """Trim only complete uniform speculative decode batches, before packing."""
    if type(k) is not int or k not in Geometry(1, max_k).ks:
        raise ValueError("K outside qualified range")
    if not counts or any(r not in drafts or counts[r] != len(drafts[r]) + 1
                         or len(drafts[r]) < k for r in counts):
        raise ValueError("cannot trim a mixed, short, or missing-draft batch")
    removed = 0
    for rid in counts:
        removed += len(drafts[rid]) - k
        counts[rid] = k + 1
        drafts[rid] = drafts[rid][:k]
    return removed
