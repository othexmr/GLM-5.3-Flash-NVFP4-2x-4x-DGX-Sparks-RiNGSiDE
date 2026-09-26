"""Bound mixed prefill work without changing cache geometry or FCFS custody.

One scheduler computes grants. Workers receive the ordinary SchedulerOutput;
there is no rank-local policy or collective decision. Pure solo prefill keeps
the original large allowance. In contention the configured cap is a TOTAL
prefill allowance per step, shared in round-robin order at existing KDA block
boundaries. This differs deliberately from the legacy per-request cap.
"""
from __future__ import annotations

from collections import deque


class FairPrefill:
    def __init__(self, block_size):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("fair prefill requires the actual scheduler block size")
        self.block_size = block_size
        self.queue = deque()
        self.grants = None
        self.decode_reserved = 0
        self.reason = "off"

    def validate_cap(self, cap):
        if cap and (cap < self.block_size or cap % self.block_size):
            raise ValueError("fair prefill cap must be a multiple of the KDA block size")

    def plan(self, scheduler, cap, is_prefilling):
        self.validate_cap(cap)
        self.grants = None
        self.decode_reserved = 0
        self.reason = "off"
        if not cap:
            self.queue.clear()
            return
        # Do not alter running-list order: the scheduler uses its tail for
        # FCFS eviction. A zero grant skips a prefill, reserving the decode
        # budget even when that decoder occurs later in the running list.
        running = list(scheduler.running)
        decoders = [r for r in running if not is_prefilling(r)]
        prefills = [r for r in running if is_prefilling(r)]
        slots = max(0, scheduler.max_num_running_reqs - len(running)
                    - scheduler.num_waiting_for_streaming_input)
        # FCFS traverses skipped_waiting before waiting. Plan all potentially
        # admitted heads up to the free slot count, including resumed heads;
        # considering only the first would misclassify two fresh prefills as
        # solo. Blocked heads are still promoted/skipped by the actual loop.
        waiting = [*scheduler.skipped_waiting, *scheduler.waiting]
        ready = [r for r in waiting
                 if getattr(r.status, "name", "") in ("WAITING", "PREEMPTED")]
        prefills.extend(ready[:slots])
        ids = {r.request_id for r in prefills}
        self.queue = deque(rid for rid in self.queue if rid in ids)
        known = set(self.queue)
        self.queue.extend(r.request_id for r in prefills if r.request_id not in known)
        if not decoders and len(prefills) <= 1:
            self.reason = "solo-prefill-original-budget"
            return
        self.reason = "contended-round-robin"
        self.grants = {rid: 0 for rid in ids}
        # Fixed K4 synchronous lane: reserve five query tokens for each
        # decoder, plus its draft input scratch reservation. This can leave
        # a few unused tokens on a joining/exiting decode, never steal them.
        self.decode_reserved = len(decoders) * (scheduler.num_spec_tokens + 1)
        spec = scheduler.vllm_config.speculative_config
        draft_slots = spec.max_num_new_slots_for_drafting
        inputs = scheduler.scheduler_config.max_num_batched_tokens
        inputs -= self.decode_reserved + len(decoders) * draft_slots
        budget = min(cap, scheduler.max_num_scheduled_tokens - self.decode_reserved)
        by_id = {r.request_id: r for r in prefills}
        for rid in self.queue:
            request = by_id[rid]
            start = request.num_computed_tokens
            end = max(request.num_prompt_tokens, request.num_tokens - 1)
            allowance = min(budget, inputs - draft_slots, max(0, end - start))
            if allowance <= 0:
                continue
            # Leave short terminal tails intact; internal chunks must end at
            # an existing KDA state boundary. Prefix-hit alignment remains the
            # scheduler's responsibility after it resolves the actual hit.
            if start + allowance < end:
                allowance = ((start + allowance) // self.block_size
                             * self.block_size - start)
            if allowance <= 0:
                continue
            self.grants[rid] = allowance
            budget -= allowance
            inputs -= allowance + draft_slots

    def limit(self, request, count):
        if self.grants is None:
            return count
        return min(count, self.grants.get(request.request_id, 0))

    def observe(self, counts):
        # A grant can fail KV/encoder admission. Yield its priority too, or a
        # blocked waiter can own the whole budget forever while the resident
        # request that could release memory receives zero tokens. This rotates
        # the offer only; computed-token/state counters remain untouched.
        offered = [rid for rid in self.queue
                   if counts.get(rid, 0) > 0 or (self.grants or {}).get(rid, 0) > 0]
        self.queue = deque(rid for rid in self.queue if rid not in offered)
        self.queue.extend(offered)
