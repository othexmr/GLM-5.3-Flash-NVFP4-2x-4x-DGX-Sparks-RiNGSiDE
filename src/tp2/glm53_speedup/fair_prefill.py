"""Bound mixed prefill work without changing cache geometry or FCFS custody.

One scheduler computes grants. Workers receive the ordinary SchedulerOutput;
there is no rank-local policy or collective decision. Pure solo prefill keeps
the original large allowance. In contention the configured cap is a TOTAL
prefill allowance per step. Short terminal tails can bypass a long prefill
once; bypassed long requests then regain round-robin priority. Internal chunks
still end at existing KDA block boundaries.
"""
from __future__ import annotations

from collections import deque


class FairPrefill:
    def __init__(self, block_size):
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("fair prefill requires the actual scheduler block size")
        self.block_size = block_size
        self.queue = deque()
        self.deferred_long = set()
        self.eligible_long = set()
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
        self.eligible_long = set()
        if not cap:
            self.queue.clear()
            self.deferred_long.clear()
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
        self.deferred_long.intersection_update(ids)
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

        def allowance_for(request, tokens, scratch):
            start = request.num_computed_tokens
            end = max(request.num_prompt_tokens, request.num_tokens - 1)
            allowance = min(tokens, scratch - draft_slots, max(0, end - start))
            if start + allowance < end:
                allowance = ((start + allowance) // self.block_size
                             * self.block_size - start)
            return max(0, allowance)

        # A terminal tail needs no new internal checkpoint. Admit it before
        # spending the entire block-sized budget on a long request. Requests
        # bypassed for a tail acquire debt below and get their next offer
        # first, so a stream of new tails cannot starve existing long work.
        tails = []
        for rid in self.queue:
            request = by_id[rid]
            remaining = max(request.num_prompt_tokens, request.num_tokens - 1) - request.num_computed_tokens
            possible = allowance_for(request, budget, inputs)
            if 0 < remaining < self.block_size and possible == remaining:
                tails.append(rid)
            elif possible > 0:
                self.eligible_long.add(rid)
        protected = [rid for rid in self.queue if rid in self.deferred_long
                     and rid in self.eligible_long]
        priority = protected + tails
        order = priority + [rid for rid in self.queue if rid not in priority]
        if tails:
            self.reason = ("contended-aged-prefill" if protected
                           else "contended-short-tail-first")
        for rid in order:
            request = by_id[rid]
            # Leave short terminal tails intact; internal chunks must end at
            # an existing KDA state boundary. Prefix-hit alignment remains the
            # scheduler's responsibility after it resolves the actual hit.
            allowance = allowance_for(request, budget, inputs)
            if allowance <= 0:
                continue
            self.grants[rid] = allowance
            budget -= allowance
            inputs -= allowance + draft_slots

    def defer_for_decode(self, decode_reserved):
        """Clear only this step's offers; preserve queue order and aging.

        The bridge skips observe() for this deliberately prefill-free
        turn. The next ordinary plan prunes departures and sees new
        waiters. No request frontier or fair priority advances here.
        """
        self.grants = {}
        self.decode_reserved = decode_reserved
        self.reason = "split-decode-repay"
        self.eligible_long = set()

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
        self.deferred_long.update(self.eligible_long - set(offered))
        # Failed KV admission yields priority as well: a request which cannot
        # allocate must not monopolise the next step's budget indefinitely.
        self.deferred_long.difference_update(offered)
        self.queue = deque(rid for rid in self.queue if rid not in offered)
        self.queue.extend(offered)
