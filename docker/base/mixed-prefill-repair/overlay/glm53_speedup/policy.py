"""CPU-only, reproducible verification-length policy; no vLLM imports."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

KS = (1, 2, 3, 4)
CONTEXTS = (8192, 32768, 131072)
# Cost captures at a nominal context include the fixed 256-token generation
# span.  This is a bounded coverage allowance, not extrapolation of an 8K curve
# to arbitrary longer contexts.
DEFAULT_OUTPUT_HEADROOM = 256


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


def decode_shapes(scope: str) -> tuple[DecodeShape, ...]:
    if scope not in ("off", "c1c2", "all"):
        raise ValueError("capture scope must be off, c1c2, or all")
    return tuple(DecodeShape(c, k) for c in range(1, 7)
                 for k in (KS if scope == "all" or scope == "c1c2" and c <= 2 else (4,)))


@dataclass(frozen=True)
class Settings:
    mode: str = "off"
    forced_k: int = 4
    scope: str = "off"

    def __post_init__(self):
        if self.mode not in ("off", "forced", "ema", "cost"):
            raise ValueError("mode must be off, forced, ema, or cost")
        if type(self.forced_k) is not int or self.forced_k not in KS:
            raise ValueError("forced_k must be an integer in 1..4")
        decode_shapes(self.scope)
        if self.mode != "off" and self.scope == "off":
            raise ValueError("adaptive modes require captured experimental shapes")


@dataclass
class RequestHistory:
    steps: int = 0
    accepted_ema: float = 4.0
    survival: list[float] = field(default_factory=lambda: [1.0] * 4)
    observations: list[int] = field(default_factory=lambda: [0] * 4)

    def observe(self, proposed: int, accepted: int):
        if type(proposed) is not int or type(accepted) is not int:
            raise ValueError("acceptance counts must be integers")
        if proposed not in KS or not 0 <= accepted <= proposed:
            raise ValueError("invalid proposed/accepted prefix")
        # The length estimator only sees full K4 draws; a smaller K right-censors
        # accepted length. Updating it as a K4 failure would lock the policy low.
        if proposed == 4:
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
    def from_document(cls, document: dict):
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


class Policy:
    def __init__(self, settings=Settings(), costs: CostTable | None = None):
        self.settings = settings
        self.costs = costs
        self.requests: dict[str, RequestHistory] = {}
        self.current_k = 4
        self.pending_k = 4
        self.pending_count = 0
        self.steps = 0
        self.members: tuple[str, ...] = ()

    def configure(self, settings: Settings):
        if settings != self.settings:
            self.settings = settings
            self.current_k = self.pending_k = 4
            self.pending_count = 0

    def retain(self, request_ids):
        live = set(request_ids)
        self.requests = {r: h for r, h in self.requests.items() if r in live}

    def observe(self, request_id, proposed, accepted):
        self.requests.setdefault(request_id, RequestHistory()).observe(proposed, accepted)

    def choose(self, request_contexts: Mapping[str, int], *, mixed=False) -> Decision:
        c = len(request_contexts)
        def result(k, reason):
            return Decision(k, reason, DecodeShape(c, k) if 1 <= c <= 6 else None)
        if self.settings.mode == "off":
            return result(4, "disabled")
        if c not in range(1, 7) or mixed:
            return result(4, "mixed_or_unsupported_batch")
        if self.settings.scope == "c1c2" and c > 2:
            return result(4, "outside_capture_scope")
        members = tuple(sorted(request_contexts))
        if members != self.members:
            self.members = members
            self.current_k = self.pending_k = 4
            self.pending_count = 0
        if self.settings.mode == "forced":
            return result(self.settings.forced_k, "forced")
        histories = [self.requests.setdefault(r, RequestHistory()) for r in members]
        self.steps += 1
        if any(h.steps < 8 for h in histories):
            return result(4, "warmup")
        if self.steps % 16 == 0:
            return result(4, "full_length_probe")
        if self.settings.mode == "ema":
            desired = max(max(1, min(4, math.ceil(h.accepted_ema + 1))) for h in histories)
        else:
            if self.costs and self.costs.execution_kind == "native_k":
                return result(4, "cost_execution_kind_mismatch")
            curve = (self.costs.curve(c, max(request_contexts.values()),
                                      include_measured_output_span=True)
                     if self.costs else None)
            if curve is None:
                return result(4, "unqualified_cost_curve")
            desired = max(KS, key=lambda k: (sum(h.expected_tokens(k) for h in histories) / curve[k], k))
        if desired == self.current_k:
            self.pending_count = 0
        else:
            self.pending_count = self.pending_count + 1 if desired == self.pending_k else 1
            self.pending_k = desired
            if self.pending_count >= 4:
                self.current_k, self.pending_count = desired, 0
        return result(self.current_k, self.settings.mode)


def trim_batch(counts: dict[str, int], drafts: dict[str, list[int]], k: int) -> int:
    """Trim only complete uniform speculative decode batches, before packing."""
    if k not in KS:
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
