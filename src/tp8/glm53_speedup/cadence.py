"""Default-off running-request cadence; admission and KV remain scheduler-owned.

One known nonfinal prefill-only offer is followed by one decode-only offer.
Fresh/waiting requests use ordinary admission first, so an unresolved prefix
hit cannot turn a presumed nonfinal chunk into a final chunk behind this gate.

Two default-off selectors for newly started decoders (decode-kernels lane, 2026-09-24). With both unset or 0, the
cadence is that of 49215137 exactly; setting both fails the boot.

P1 first-token repay, GLM53_SPEEDUP_FIRST_TOKEN_REPAY=1:
- A request that has just become a decoder is owed ONE decode-only offer before the next prefill-only offer. "Just
  become" means its prompt finished and it produced its first token. Without the repay, its first decode waits a whole
  prefill-only chunk (0.55-1.0 s on TP4) before the repay turn.
- The debt only converts a turn that would otherwise be a prefill-only offer. A mixed turn already decodes every
  decoder and clears the debt; a prefill-only turn never does.
- The debt costs at most ONE decode-only step per wave of newly started decoders. A converted turn clears every current
  debt, whether or not the decoder got tokens, so the debt can never stall the prefill.

P2 first-content burst, GLM53_SPEEDUP_FIRST_CONTENT_BURST=N (1 <= N <= 16):
- A newly started decoder whose prompt ends in <think> is owed while it has produced no token after </think>, i.e. no
  visible content yet. Token ids are those of the GLM-5.3-Flash tokenizer (THINK_START / THINK_END below).
- While any owed decoder still has no content, a turn that would be a prefill-only offer becomes a decode-only turn.
- Two caps bound the cost:
  - a decoder counts every converted turn and leaves the owed set after N of them;
  - at most N converted turns run in a row between two prefill-only offers.
- A decoder whose prompt does not end in <think> (completions, thinking disabled) is never owed: its first token is
  already visible.

Both: owed_decode (the repay after a prefill-only offer, which also lets a decoder whose completion can release memory
run after a failed prefill offer) is untouched and takes precedence. Neither selector changes admission, grants, KV or
request counters. Both only choose which already-eligible turn is decode-only.
"""
import os
import re

THINK_START = 154841   # '<think>' in the GLM-5.3-Flash tokenizer; the chat template's generation prompt ends with it
THINK_END = 154842     # '</think>'
_MAX_BURST = 16

_FIRST_TOKEN_REPAY = os.environ.get("GLM53_SPEEDUP_FIRST_TOKEN_REPAY", "0").strip() or "0"
if _FIRST_TOKEN_REPAY not in ("0", "1"):
    raise ValueError("invalid first-token repay selector")
_BURST_RAW = os.environ.get("GLM53_SPEEDUP_FIRST_CONTENT_BURST", "0").strip() or "0"
if not re.fullmatch(r"[0-9]{1,2}", _BURST_RAW) or int(_BURST_RAW) > _MAX_BURST:
    raise ValueError("invalid first-content burst selector (0 off, 1-16 cap)")
_FIRST_CONTENT_BURST = int(_BURST_RAW)
if _FIRST_CONTENT_BURST and _FIRST_TOKEN_REPAY == "1":
    raise ValueError("first-token repay and first-content burst are exclusive selectors")


def _thinking(request):
    ids = getattr(request, 'prompt_token_ids', None)
    return bool(ids) and ids[-1] == THINK_START


def _has_content(request):
    """True once the request produced a token after its first </think>."""
    out = request.output_token_ids
    for i, token in enumerate(out):
        if token == THINK_END:
            return i < len(out) - 1
    return False


class SplitCadence:
    def __init__(self):
        self.owed_decode = set()
        self.allowed = None
        self.phase = 'mixed-fallback'
        self.reason = 'initial'
        self.decoder_ids = set()
        self.prefill_id = None
        self.prefill_start = self.prefill_end = None
        self.first_token_repay = _FIRST_TOKEN_REPAY == "1"
        self.first_content_burst = _FIRST_CONTENT_BURST
        self.known_decoders = set()
        self.first_debt = set()
        self.burst_owed = {}
        self.burst_run = 0

    def _track_new_decoders(self, running, is_prefilling):
        new = self.decoder_ids - self.known_decoders
        self.known_decoders = set(self.decoder_ids)
        if self.first_token_repay:
            # A decoder not seen as one at the previous turn has just produced its first token.
            self.first_debt.intersection_update(self.decoder_ids)
            self.first_debt.update(new)
        if self.first_content_burst:
            by_id = {r.request_id: r for r in running}
            for rid in list(self.burst_owed):
                if (rid not in self.decoder_ids or self.burst_owed[rid] >= self.first_content_burst
                        or _has_content(by_id[rid])):
                    del self.burst_owed[rid]
            for rid in new:
                if _thinking(by_id[rid]) and not _has_content(by_id[rid]):
                    self.burst_owed[rid] = 0
            if not any(is_prefilling(r) for r in running):
                self.burst_run = 0     # no running prefill to protect: a later one starts a fresh run

    def begin(self, scheduler, cap, is_prefilling):
        self.allowed = None
        self.phase = 'mixed-fallback'
        self.reason = 'no-eligible-running-pair'
        self.prefill_id = None
        self.prefill_start = self.prefill_end = None
        running = scheduler.running
        self.decoder_ids = {r.request_id for r in running if not is_prefilling(r)}
        self.owed_decode.intersection_update(self.decoder_ids)
        if self.first_token_repay or self.first_content_burst:
            self._track_new_decoders(running, is_prefilling)
        # A failed prefill offer must also yield to the decoder whose completion
        # can release memory. New waiters can wait this one turn, then admission
        # returns to the unchanged policy. Never pre-advance request counters.
        if self.owed_decode:
            self.phase = 'decode'
            self.reason = 'repay-prefill-offer'
            self.allowed = set(self.decoder_ids)
            return
        if not self.decoder_ids or scheduler.waiting or scheduler.skipped_waiting:
            return
        prefills = [r for r in running if is_prefilling(r)]
        if len(prefills) != 1:
            return
        request = prefills[0]
        # Limit this first packet to an already admitted original prompt. Replay,
        # multimodal, streaming and in-flight output need separate qualification.
        if (getattr(request.status, 'name', '') != 'RUNNING'
                or request.num_tokens != request.num_prompt_tokens
                or request.num_computed_tokens <= 0
                or request.num_computed_tokens + cap >= request.num_prompt_tokens
                or any(getattr(r, 'has_encoder_inputs', False)
                       or getattr(r, 'num_output_placeholders', 0)
                       or getattr(r, 'num_stale_output_tokens', 0)
                       for r in running)):
            return
        # This turn would be a prefill-only offer. A newly started decoder may take it first (decode-only).
        if self.first_debt:
            self.phase = 'decode'
            self.reason = 'first-token-repay'
            self.allowed = set(self.decoder_ids)
            return
        if self.burst_owed and self.burst_run < self.first_content_burst:
            self.phase = 'decode'
            self.reason = 'first-content-burst'
            self.allowed = set(self.decoder_ids)
            return
        self.prefill_id = request.request_id
        self.prefill_start = request.num_computed_tokens
        self.prefill_end = request.num_prompt_tokens
        self.phase = 'prefill'
        self.reason = 'one-known-nonfinal-running-prefill'
        self.allowed = {self.prefill_id}

    def limit(self, request, count):
        return count if self.allowed is None or request.request_id in self.allowed else 0

    def deferred(self, request):
        return self.allowed is not None and request.request_id not in self.allowed

    def observe(self, counts):
        if self.phase == 'prefill':
            self.owed_decode = set(self.decoder_ids)
        elif self.phase == 'decode':
            self.owed_decode.difference_update(rid for rid, count in counts.items() if count > 0)
        if self.first_token_repay:
            if self.reason == 'first-token-repay':
                # One converted turn settles every current debt, scheduled or not: the debt can never hold the
                # prefill back for more than one step per wave of new decoders.
                self.first_debt.clear()
            elif self.phase != 'prefill':
                # A decode-only or mixed turn that scheduled the decoder settles its first-token debt.
                self.first_debt.difference_update(
                    rid for rid, count in counts.items() if count > 0 and rid in self.decoder_ids)
        if self.first_content_burst:
            if self.reason == 'first-content-burst':
                self.burst_run += 1
                for rid in self.burst_owed:
                    self.burst_owed[rid] += 1
            elif self.phase == 'prefill':
                self.burst_run = 0

    def snapshot(self):
        out = dict(phase=self.phase, reason=self.reason, prefill_id=self.prefill_id,
                   prefill_start=self.prefill_start, prefill_end=self.prefill_end,
                   decoder_ids=sorted(self.decoder_ids),
                   allowed=sorted(self.allowed) if self.allowed is not None else None,
                   owed_decode=sorted(self.owed_decode))
        if self.first_token_repay:
            out['first_debt'] = sorted(self.first_debt)
        if self.first_content_burst:
            out['burst_owed'] = dict(sorted(self.burst_owed.items()))
            out['burst_run'] = self.burst_run
        return out
