"""Explicit target descriptors; the DFlash2 drafter keeps its configured proposal length."""
import json
import logging
import os
from .policy import decode_shapes
from .geometry import Geometry, capture_axes_from_env

 # Route the opt-in stream through vLLM's existing INFO handler.  The image's
 # root logger has no handler and the package logger propagates to it, so an
 # ordinary module logger silently drops INFO records despite VLLM_LOGGING_LEVEL.
log = logging.getLogger("vllm.glm53_speedup.graphs")
# Keep graph-memory/descriptors visible for the opt-in qualification stream.
log.setLevel(logging.INFO)


def eligible_descriptor(desc, requests, tokens, query, max_query):
    if (os.environ.get("GLM53_SPEEDUP_PREFILL_TAILS", "0") == "1"
            and getattr(desc.cg_mode, "name", str(desc.cg_mode)) == "PIECEWISE" and desc.num_tokens in (64, 128)):
        # A tail graph is only safe for one request whose actual scheduled
        # length and query geometry both match the captured descriptor.  The
        # graph selector may pass ``None`` for a non-uniform query summary;
        # that remains admissible only when the exact max-query guard below
        # still closes the geometry.  A reported query length must agree too,
        # otherwise a 64/128 graph can be selected for a different tail.
        return (requests == 1 and tokens == desc.num_tokens
                and max_query == tokens
                and (query is None or query == desc.num_tokens))
    return True


def capture_start(manager):
    if os.environ.get("GLM53_SPEEDUP_GRAPH_MEMORY", "0") != "1":
        return
    import torch
    torch.cuda.synchronize()
    manager._speedup_memory_before = (torch.cuda.mem_get_info()[0],
                                     torch.cuda.memory_allocated(), torch.cuda.memory_reserved())


def capture_end(manager):
    if not hasattr(manager, "_speedup_memory_before"):
        return
    import torch
    torch.cuda.synchronize()
    before = manager._speedup_memory_before
    after = (torch.cuda.mem_get_info()[0], torch.cuda.memory_allocated(), torch.cuda.memory_reserved())
    log.info("GLM53_SPEEDUP_GRAPH_MEMORY %s", json.dumps({
        "rank": torch.distributed.get_rank(), "manager": type(manager).__name__,
        "profiling_sample": manager._max_full_descs_to_capture is not None,
        "free_before": before[0], "free_after": after[0], "free_delta": before[0] - after[0],
        "allocated_delta": after[1] - before[1], "reserved_delta": after[2] - before[2],
        "captured_full_descriptors": [str(d) for d in manager.graphs],
        "planned_descriptors": [str(d) for ds in manager._capture_descs.values() for d in ds],
    }, sort_keys=True))


def extend_candidates(manager, by_mode, descriptor, mode):
    # DFlashCudaGraphManager receives the target config too. Identify the actual
    # runner, not its config, to avoid capturing a shorter fixed-layout drafter.
    if type(manager).__name__ != "ModelCudaGraphManager":
        return
    scope = os.environ.get("GLM53_SPEEDUP_CAPTURE", "off")
    tails = os.environ.get("GLM53_SPEEDUP_PREFILL_TAILS", "0") == "1"
    if scope == "off" and not tails:
        return
    joint = os.environ.get("GLM53_SPEEDUP_FAIR_COST_JOINT", "0")
    if joint not in ("0", "1"):
        raise ValueError("invalid joint fair/Cost-K selector")
    # CAPTURE is already opt-in. Both qualified target topologies use
    # the same (C, K+1) descriptor geometry; the drafter retains the configured proposal length.
    config = getattr(manager, "vllm_config", None)
    proposal_k = getattr(config, "num_speculative_tokens", manager.decode_query_len - 1)
    if manager.decode_query_len != proposal_k + 1:
        raise ValueError("target decode query length must be configured K plus one")
    geometry = Geometry.runtime(manager.max_num_reqs, proposal_k)
    if manager.tp_size not in (2, 4):
        raise ValueError("experimental graph geometry requires TP2 or TP4")
    if joint == "1" and (manager.tp_size != 4 or geometry != Geometry()):
        raise ValueError("joint fair/Cost-K requires TP4")
    if joint == "1" and (scope != "all" or tails):
        raise ValueError("joint TP4 Cost-K requires full decode scope without prefill-tail changes")
    if manager.varlen_decode:
        raise ValueError("adaptive-K requires uniform target graphs")
    if scope != "off":
        if manager.cudagraph_mode.decode_mode() != mode.FULL:
            raise ValueError("adaptive-K requires FULL target decode graphs")
        requests, ks = capture_axes_from_env(geometry.max_requests, geometry.max_k)
        for shape in decode_shapes(scope, geometry.max_requests, geometry.max_k, requests=requests, ks=ks):
            d = descriptor(cg_mode=mode.FULL, num_tokens=shape.tokens,
                           num_reqs=shape.concurrency, uniform_token_count=shape.query_len)
            if d not in by_mode[mode.FULL]:
                by_mode[mode.FULL].append(d)
    if tails:
        if manager.cudagraph_mode.mixed_mode() != mode.PIECEWISE:
            raise ValueError("prefill tails require FULL_AND_PIECEWISE with breakable graphs")
        if not manager.use_breakable_cg:
            raise ValueError("prefill tails require breakable graph support")
        for n in (64, 128):
            d = descriptor(cg_mode=mode.PIECEWISE, num_tokens=n,
                           num_reqs=1, max_query_len=n)
            if d not in by_mode[mode.PIECEWISE]:
                by_mode[mode.PIECEWISE].append(d)
    log.info("GLM53_SPEEDUP_GEOMETRY selected=True consumer=target_graphs requests=%d proposal_k=%d", geometry.max_requests, geometry.max_k)
    log.info("GLM53_SPEEDUP_GRAPH_DESCRIPTORS %s", json.dumps([
        {"mode": str(m), "tokens": d.num_tokens, "requests": d.num_reqs,
         "query": d.uniform_token_count, "max_query": d.max_query_len}
        for m, ds in by_mode.items() for d in ds], sort_keys=True))
