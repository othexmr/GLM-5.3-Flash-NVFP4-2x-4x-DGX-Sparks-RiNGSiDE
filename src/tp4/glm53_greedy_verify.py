"""Greedy DFlash2 verification without gathering the full logits (decode-kernels lane, 2026-09-24; default off).

Today, every verify step computes the lm_head shard on each TP rank and all-gathers the full [M, 154,880] bf16 logits
to every rank (LogitsProcessor._gather_logits). Then the rejection sampler reads every row block by block
(_compute_local_logits_stats_kernel, 8,192-column blocks; the bonus row in _resample_kernel, 1,024-column blocks).
For a temperature-0 request, all it uses is each row's argmax:
- the lowest index among equal maxima (tl.max tie-break-left in a block, tl.argmax over blocks);
- the bf16 values upcast to fp32;
- NaN block maxima mapped to -inf.

GLM53_GREEDY_ARGMAX_VERIFY=1 replaces this for a batch in which every request is greedy, and nothing else reads the
logits. Each rank computes its shard's (max, lowest global index), with NaN taken as -inf and columns past the
shard's real vocabulary excluded. It all-gathers only those (value, index) pairs: [M, 2] fp32 per rank; indices
below 2^24 are exact in fp32. The global argmax takes the lowest rank on equal values, so the lowest global index.
The greedy acceptance of the rejection kernels then runs on those argmaxes:
- accept a draft iff it equals the target argmax and is not the -1 placeholder;
- on the first rejection, store the target argmax;
- after all drafts are accepted, store the bonus row's argmax;
- num_sampled = accepted + 1, then vLLM's get_num_sampled_and_rejected as before.

Eligibility is checked per batch on the host; any batch that fails takes today's path unchanged. A batch is eligible
when all of these hold:
- no batch sharder;
- no grammar or structured output this step;
- drafts present, and a greedy drafter (draft_logits is None);
- standard rejection sampling (not synthetic, not block verification, no adaptive verification);
- no NaN counting, no trace replay, no sampling mask;
- no logprobs or logprob token ids in the batch;
- every request's temperature is exactly 0.0;
- no request needs logits processing (logit bias, allowed ids, min_tokens, penalties, bad words, thinking budget,
  min_p, top_k, top_p);
- the head is a plain LogitsProcessor (scale 1, no soft cap) whose lm_head shards the vocabulary over the TP group.

Exactness:
- Finite and +-inf logits: identical token choices by construction (the same values, the same maximum, the same
  lowest-index rule).
- NaN: today maps a NaN block maximum to -inf, but inside a block the tl.max(return_indices) reduction order decides
  whether a NaN displaces the block's finite maximum. This path ignores NaN elements. Rows with NaN may therefore
  differ; the leaf measures that case separately (README "Leaf 2 rule").
Only temperature-0 requests benefit; RigMark runs greedy (temperature 0.0, top_p 1.0)."""
import os
import sys

import numpy as np
import torch

from vllm.triton_utils import tl, triton

ENABLED = os.environ.get("GLM53_GREEDY_ARGMAX_VERIFY", "").strip() == "1"
LOG_EVERY = int(os.environ.get("GLM53_GREEDY_ARGMAX_LOG_EVERY", "").strip() or "0")
_STATE = {"engaged": False, "fast": 0, "fallback": 0, "head": None, "why": None}


@triton.jit
def _local_argmax_kernel(logits_ptr, stride, n_real, vocab_start, out_ptr, BLOCK: tl.constexpr):
    # One program per row: (max, lowest index of the max) over the shard's real columns; NaN counts as -inf.
    row = tl.program_id(0).to(tl.int64)
    best_v = tl.full((), float("-inf"), tl.float32)
    best_i = tl.full((), 0, tl.int32)
    for start in range(0, n_real, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        m = cols < n_real
        x = tl.load(logits_ptr + row * stride + cols, mask=m, other=float("-inf")).to(tl.float32)
        x = tl.where(x != x, float("-inf"), x)
        v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
        take = v > best_v                      # strictly greater: an earlier block keeps its lower index on ties
        best_i = tl.where(take, start + i, best_i)
        best_v = tl.where(take, v, best_v)
    tl.store(out_ptr + row * 2, best_v)
    tl.store(out_ptr + row * 2 + 1, (best_i + vocab_start).to(tl.float32))


@triton.jit
def _greedy_accept_kernel(sampled_ptr, sampled_stride, num_sampled_ptr, argmax_ptr, draft_ptr, cu_num_logits_ptr):
    # The greedy branch of _rejection_kernel + _insert_resampled_kernel on precomputed target argmaxes.
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(num_draft_tokens):
        logit_idx = start_idx + i
        draft_sampled = tl.load(draft_ptr + logit_idx + 1).to(tl.int64)
        is_valid_draft = draft_sampled >= 0
        draft_sampled = tl.maximum(0, draft_sampled)
        if verifying:
            target_argmax = tl.load(argmax_ptr + logit_idx)
            accepted = (target_argmax == draft_sampled) & is_valid_draft
            verifying = accepted
            accepted_length += accepted
            tl.store(sampled_ptr + req_idx * sampled_stride + i, tl.where(accepted, draft_sampled, target_argmax))
    # Rejected at accepted_length (target argmax already stored there) or all accepted (the bonus row's argmax).
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, tl.load(argmax_ptr + start_idx + accepted_length))
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


def local_pairs(local_logits: torch.Tensor, n_real: int, vocab_start: int) -> torch.Tensor:
    """[M, 2] fp32: (shard max with NaN as -inf, lowest global index of it) per row."""
    assert local_logits.ndim == 2 and local_logits.stride(1) == 1 and 0 < n_real <= local_logits.shape[1]
    rows = local_logits.shape[0]
    out = torch.empty((rows, 2), dtype=torch.float32, device=local_logits.device)
    if rows:
        _local_argmax_kernel[(rows,)](local_logits, local_logits.stride(0), n_real, vocab_start, out,
                                      BLOCK=4096, num_warps=8)
    return out


def combine_pairs(gathered: torch.Tensor) -> torch.Tensor:
    """[M, tp, 2] (rank order = vocab order) -> [M] int64 global argmax; equal values go to the lowest rank."""
    best_rank = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered[:, :, 1].gather(-1, best_rank).squeeze(-1).to(torch.int64)


def greedy_accept(target_argmax: torch.Tensor, draft_sampled: torch.Tensor, cu_num_logits: torch.Tensor,
                  num_reqs: int, num_speculative_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    sampled = torch.empty((num_reqs, num_speculative_steps + 1), dtype=torch.int64, device=target_argmax.device)
    num_sampled = torch.empty((num_reqs,), dtype=torch.int32, device=target_argmax.device)
    _greedy_accept_kernel[(num_reqs,)](sampled, sampled.stride(0), num_sampled, target_argmax.contiguous(),
                                       draft_sampled.contiguous(), cu_num_logits, num_warps=1)
    return sampled, num_sampled


def _head(runner):
    if _STATE["head"] is None:
        model, found = runner.model, None
        for m in (model, getattr(model, "language_model", None), getattr(getattr(model, "model", None), "language_model", None)):
            if m is not None and hasattr(m, "logits_processor") and hasattr(m, "lm_head"):
                found = (m.logits_processor, m.lm_head)
                break
        ok = found is not None
        if ok:
            lp, head = found
            from vllm.distributed import get_tensor_model_parallel_world_size
            si = getattr(head, "shard_indices", None)
            ok = (getattr(lp, "scale", None) == 1.0 and getattr(lp, "soft_cap", 1) is None
                  and not getattr(lp, "logits_as_input", True) and si is not None
                  and getattr(head, "tp_size", None) == get_tensor_model_parallel_world_size()
                  and si.org_vocab_end_index > si.org_vocab_start_index)
        _STATE["head"] = found if ok else False
        if not ok:
            print("GLM53_GREEDY_ARGMAX_VERIFY: head not eligible, today's path everywhere", file=sys.stderr, flush=True)
    return _STATE["head"] or None


def eligible(runner, input_batch, grammar_output) -> bool:
    """Per-batch host check; False keeps today's path for this batch."""
    try:
        idx = input_batch.idx_mapping_np
        sampler, rs, spec = runner.sampler, runner.rejection_sampler, runner.speculator
        ok = (grammar_output is None and runner.batch_sharder is None and input_batch.num_reqs > 0
              and input_batch.num_draft_tokens > 0 and rs is not None and spec is not None and sampler is not None
              and getattr(spec, "draft_logits", None) is None
              and rs.synthetic_conditional_rates is None and not rs.use_block_verification
              and not getattr(rs, "enable_adaptive_verification", False)
              and getattr(runner, "adaptive_verification", None) is None
              and not sampler.compute_nans and getattr(sampler, "trace_replay_state", None) is None
              and not getattr(sampler, "return_sampling_mask", False)
              and sampler.get_logprobs_dims(idx) is None
              and bool(np.all(sampler.sampling_states.temperature.np[idx] == 0.0))
              and not bool(np.any(sampler.needs_logits_processing[idx]))
              and _head(runner) is not None)
    except Exception as exc:  # any unexpected shape of the runner: today's path, reported once
        if _STATE["why"] is None:
            _STATE["why"] = repr(exc)
            print(f"GLM53_GREEDY_ARGMAX_VERIFY: eligibility check failed ({exc!r}); today's path", file=sys.stderr, flush=True)
        ok = False
    if not ok:
        _STATE["fallback"] += 1
    return ok


def verify(runner, sample_hidden_states, input_batch):
    """Greedy verification from (value, index) pairs; returns the SamplerOutput the rejection sampler would."""
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    lp, head = _head(runner)
    local = lp._apply_head(head, sample_hidden_states, None)
    si = head.shard_indices
    pairs = local_pairs(local, si.org_vocab_end_index - si.org_vocab_start_index, si.org_vocab_start_index)
    tp = head.tp_size
    gathered = tensor_model_parallel_all_gather(pairs, dim=-1) if tp > 1 else pairs
    target_argmax = combine_pairs(gathered.reshape(pairs.shape[0], tp, 2))
    draft_sampled = input_batch.input_ids[input_batch.logits_indices]
    sampled, num_sampled = greedy_accept(target_argmax, draft_sampled, input_batch.cu_num_logits,
                                         input_batch.num_reqs, runner.rejection_sampler.num_speculative_steps)
    num_sampled, num_rejected = get_num_sampled_and_rejected(num_sampled, input_batch.seq_lens,
                                                             input_batch.cu_num_logits, input_batch.idx_mapping,
                                                             runner.sampler.req_states.prefill_len.gpu)
    _STATE["fast"] += 1
    if not _STATE["engaged"]:
        _STATE["engaged"] = True
        print(f"GLM53_GREEDY_ARGMAX_VERIFY engaged: tp {tp}, shard [{si.org_vocab_start_index}, "
              f"{si.org_vocab_end_index}), pairs all-gathered instead of logits", file=sys.stderr, flush=True)
    if LOG_EVERY and (_STATE["fast"] + _STATE["fallback"]) % LOG_EVERY == 0:
        print(f"GLM53_GREEDY_ARGMAX_VERIFY steps fast={_STATE['fast']} fallback={_STATE['fallback']}",
              file=sys.stderr, flush=True)
    return SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                         num_sampled=num_sampled, num_rejected=num_rejected)
