"""Opt-in CUDA event timing, drained without adding synchronization to decode."""
from collections import deque
import json
import logging
import os

_enabled = os.environ.get("GLM53_SPEEDUP_TIMING", "0") == "1"
_pending = deque()
_batch = {}
 # Route the opt-in stream through vLLM's existing INFO handler.  The image's
 # root logger has no handler and the package logger propagates to it, so an
 # ordinary module logger silently drops INFO records despite VLLM_LOGGING_LEVEL.
log = logging.getLogger("vllm.glm53_speedup.telemetry")
# Timing records are an opt-in measurement stream; make the module's explicit
# INFO records observable without changing global logging or decode sync.
log.setLevel(logging.INFO)


def enabled():
    return _enabled


def record_batch(batch, desc):
    global _batch
    if not _enabled:
        return
    _batch = {"requests": batch.num_reqs, "tokens": batch.num_tokens,
              "query_lengths": batch.num_scheduled_tokens.tolist(),
              "context_lengths": batch.num_computed_tokens_np.tolist(),
              "prefill": batch.is_prefilling_np.tolist(),
              "descriptor": {"mode": getattr(desc.cg_mode, "name", str(desc.cg_mode)), "requests": desc.num_reqs,
                             "tokens": desc.num_tokens, "query": desc.uniform_token_count,
                             "max_query": desc.max_query_len},
              "padding": batch.num_tokens_after_padding - batch.num_tokens}
    flush()


def flush():
    while _pending and _pending[0][0].drafter_end.query():
        events, batch = _pending.popleft()
        log.info("GLM53_SPEEDUP_TIMING %s", json.dumps(dict(batch,
            target_ms=events.forward_start.elapsed_time(events.forward_end),
            draft_ms=events.drafter_start.elapsed_time(events.drafter_end),
            gpu_step_ms=events.forward_start.elapsed_time(events.drafter_end)), sort_keys=True))


def drain(events):
    flush()
    if len(_pending) >= 1024:
        raise RuntimeError("GPU timing backlog exceeded 1024: measurement invalid")
    _pending.append((events, dict(_batch)))
