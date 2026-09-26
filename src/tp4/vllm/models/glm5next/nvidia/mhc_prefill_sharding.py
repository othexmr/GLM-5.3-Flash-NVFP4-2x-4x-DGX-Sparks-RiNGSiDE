# SPDX-License-Identifier: Apache-2.0
# Adapted from FujitsuPolycom/SparkRing 61f277bd.
# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): NVFP4 port with explicit all-rank admission; admission widened to
# eager forwards of at least GLM53_MHC_PREFILL_MIN_ROWS rows, mixed batches included; opt-in RDMA ring for the owner
# collectives (GLM53_MHC_PREFILL_RDMA).
# NVFP4 port: local KDA/MoE sources, explicit all-rank admission.
# v8 (2026-09-23): admission widened from pure-prefill 4608/13824-row forwards to every eager target forward
# of at least GLM53_MHC_PREFILL_MIN_ROWS rows, mixed prefill+decode batches included. A row count that is not a
# multiple of four is zero-padded inside the owner collectives only: attention, MoE, the drafter taps and the final
# hidden states see exactly the real rows, and the padding rows' mHC outputs are dropped at every all-gather.
# v10 (rdma-prefill lane, 2026-09-24): opt-in GLM53_MHC_PREFILL_RDMA=1 routes every owner reduce-scatter and
# all-gather of at least GLM53_MHC_PREFILL_RDMA_MIN_ROWS rows through our RDMA prefill ring (glm53_prefill_rdma: NCCL's
# ring order, bit-identical results) instead of PyNccl. Default off; smaller forwards keep PyNccl.
"""Opt-in, eager TP4 prefill ownership. No persistent recurrent or decode state."""
from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

MAX_ROWS = 14336            # the qualified scheduler token budget
MIN_ROWS_DEFAULT = 1024
MIN_ROWS_ENV = 'GLM53_MHC_PREFILL_MIN_ROWS'
HIDDEN = 4096
SELECTOR = 'GLM53_MHC_PREFILL_SHARD'
RDMA_ENV = 'GLM53_MHC_PREFILL_RDMA'
RDMA_MIN_ROWS_ENV = 'GLM53_MHC_PREFILL_RDMA_MIN_ROWS'
_LOG = logging.getLogger(__name__)
_REPORTS: dict[tuple, int] = {}
_TOTALS = {'engaged': 0, 'engaged_rows': 0, 'padded': 0, 'fallback_at_or_above_threshold': 0, 'rdma': 0}


def parse_min_rows(raw: str) -> int:
    value = int(raw)
    if not 1 <= value <= MAX_ROWS:
        raise RuntimeError(f'{MIN_ROWS_ENV} must be 1..{MAX_ROWS}')
    return value


def validate_config(config, min_rows):
    p = config.parallel_config
    if (p.tensor_parallel_size != 4 or p.decode_context_parallel_size != 1
            or p.pipeline_parallel_size != 1 or p.data_parallel_size != 1
            or p.prefill_context_parallel_size != 1 or p.enable_expert_parallel
            or p.enable_eplb or p.use_sequence_parallel_moe):
        raise RuntimeError('mHC sharding requires TP4/DCP1/PP1/DP1/PCP1 without EP/EPLB/SP')
    if config.scheduler_config.max_num_batched_tokens != MAX_ROWS:
        raise RuntimeError('mHC packet requires the unchanged 14336-token scheduler budget')
    if config.scheduler_config.async_scheduling:
        raise RuntimeError('mHC packet requires the qualified synchronous scheduler')
    sizes = config.compilation_config.cudagraph_capture_sizes or []
    if any(size >= min_rows for size in sizes):
        raise RuntimeError('mHC ownership threshold must exceed every CUDA graph capture size')


def configure(model: Any, config: Any, env: Any) -> None:
    import torch
    from vllm.distributed import get_tp_group
    raw = env.get(SELECTOR, '0')
    raw_min = env.get(MIN_ROWS_ENV, str(MIN_ROWS_DEFAULT))
    raw_rdma = env.get(RDMA_ENV, '0')
    raw_rdma_min = env.get(RDMA_MIN_ROWS_ENV, raw_min)
    error, min_rows, rdma_min = None, None, None
    try:
        if raw not in ('0', '1'):
            raise RuntimeError('invalid ' + SELECTOR)
        if raw_rdma not in ('0', '1'):
            raise RuntimeError('invalid ' + RDMA_ENV)
        min_rows = parse_min_rows(raw_min)
        rdma_min = parse_min_rows(raw_rdma_min)
        if raw_rdma == '1' and (raw != '1' or rdma_min < min_rows):
            raise RuntimeError(f'{RDMA_ENV} needs {SELECTOR}=1 and {RDMA_MIN_ROWS_ENV} >= {MIN_ROWS_ENV}')
        if raw == '1':
            validate_config(config, min_rows)
    except (AttributeError, RuntimeError, ValueError) as exc:
        error = str(exc)
    group = get_tp_group()
    mine = (raw, raw_min, raw_rdma, raw_rdma_min)
    votes = [None] * group.world_size
    torch.distributed.all_gather_object(votes, mine + (error,), group=group.cpu_group)
    if any(v != mine + (None,) for v in votes):
        raise RuntimeError('mHC configuration disagreement/error: ' + repr(votes))
    model._mhc_prefill_enabled = raw == '1'
    model._mhc_prefill_min_rows = min_rows
    model._mhc_prefill_config = config
    model._mhc_prefill_rdma = None
    model._mhc_prefill_rdma_min_rows = rdma_min
    if model._mhc_prefill_enabled:
        _LOG.warning('GLM53_MHC_PREFILL_READY rank=%d min_rows=%d max_rows=%d mixed=1 pad=4 tp=4 default_off=1',
                     group.rank_in_group, min_rows, MAX_ROWS)
    if model._mhc_prefill_enabled and raw_rdma == '1':
        # Every rank reaches this point together (the vote above). PrefillRing's construction is itself collective over
        # the TP CPU group and voted at every stage, so either all four ranks have the ring or all four raise.
        from glm53_prefill_rdma.ring import PrefillRing
        comm = getattr(group.device_communicator, 'pynccl_comm', None)
        if comm is None or comm.world_size != 4 or comm.rank != group.rank_in_group:
            raise RuntimeError('mHC RDMA collectives require the active TP4 PyNccl communicator')
        model._mhc_prefill_rdma = PrefillRing(group.cpu_group, group.rank_in_group, comm.device, max_rows=MAX_ROWS)
        _LOG.warning('GLM53_MHC_PREFILL_RDMA_READY rank=%d min_rows=%d', group.rank_in_group, rdma_min)


def batch_metadata(metadata: Any, names: tuple[str, ...], rows: int) -> str | None:
    """None when every recurrent-attention metadata owner describes exactly `rows` real tokens, else a reason.

    Prefill, decode and speculative-decode tokens are all admitted: ownership only moves the row-wise mHC maps
    between two collectives; attention and MoE still run on the full, unpadded batch.
    """
    if not isinstance(metadata, dict) or not names:
        return 'no-metadata'
    for name in names:
        item = metadata.get(name)
        values = [getattr(item, f, None) for f in
                  ('num_prefill_tokens', 'num_decode_tokens', 'num_spec_decode_tokens', 'num_actual_tokens')]
        if any(type(v) is not int for v in values):
            return 'metadata-type'
        prefill, decode, spec, actual = values
        if actual != rows or prefill + decode + spec != rows:
            return f'metadata-rows p={prefill} d={decode} s={spec} n={actual}'
    return None


def agree_admission(votes):
    """A disagreement is fatal before any deferred projection is computed."""
    if len(votes) != 4 or any(not isinstance(v, tuple) or len(v) != 3 for v in votes):
        raise RuntimeError('invalid TP4 mHC admission vote')
    if any(v[2] is not None for v in votes):
        raise RuntimeError('mHC capability failure: ' + repr(votes))
    if len({(v[0], v[1]) for v in votes}) != 1:
        raise RuntimeError('mHC rank admission disagreement: ' + repr(votes))
    return votes[0][0]


def validate_moe_deferral(runner: Any) -> None:
    c = runner.moe_config
    if (c.tp_size != 4 or c.dp_size != 1 or c.ep_size != 1 or c.pcp_size != 1
            or c.is_sequence_parallel or c.skip_final_all_reduce
            or c.moe_parallel_config.use_all2all_kernels or runner._fused_output_is_reduced
            or runner.routed_output_transform is not None or runner.routed_input_transform is not None
            or type(runner.router).__name__ == 'ZeroExpertRouter'):
        raise RuntimeError('mHC requires an unreduced conventional TP4 MoE result')


def validate_model(model: Any) -> tuple[str, ...]:
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from .attention import Glm5NextMLAAttention
    from .kda import Glm5NextLinearAttention
    validate_config(model._mhc_prefill_config, model._mhc_prefill_min_rows)
    if model.is_sequence_parallel or len(model._active_layers) != 45:
        raise RuntimeError('mHC packet requires the complete 45-layer target stack')
    names = []
    for index, layer in enumerate(model._active_layers):
        if (not layer.mhc or layer.is_mtp_layer or layer.layer_idx != index
                or layer.num_hidden_layers != 45 or layer.n != 4):
            raise RuntimeError('mHC requires uniform, unmodified target-layer ownership')
        attn = layer.self_attn
        if type(attn) is Glm5NextLinearAttention:
            names.append(attn.prefix)
        elif type(attn) is Glm5NextMLAAttention:
            if type(attn.mla_attn) is not MultiHeadLatentAttentionWrapper:
                raise RuntimeError('unqualified MLA wrapper')
        else:
            raise RuntimeError('unqualified attention implementation')
        projection = attn.o_proj
        if (type(projection) is not RowParallelLinear or projection.tp_size != 4
                or not projection.reduce_results or projection.bias is not None):
            raise RuntimeError('unqualified attention output reduction')
        if layer._mlp_is_moe:
            if type(layer.mlp.experts) is not MoERunner:
                raise RuntimeError('unqualified MoE runner')
            validate_moe_deferral(layer.mlp.experts)
        else:
            projection = layer.mlp.down_proj
            if (type(projection) is not RowParallelLinear or projection.tp_size != 4
                    or not projection.reduce_results or projection.bias is not None):
                raise RuntimeError('unqualified dense FFN output reduction')
    if not names:
        raise RuntimeError('missing recurrent attention metadata owners')
    return tuple(names)


@dataclass
class PrefillOwnership:
    comm: Any
    rank: int
    rows: int
    rs_count: int = 0
    ag_count: int = 0
    rdma: Any = None
    owner_rows: int = field(init=False)
    padded_rows: int = field(init=False)
    valid_rows: int = field(init=False)

    def __post_init__(self):
        if not 1 <= self.rows <= MAX_ROWS or self.rank not in range(4):
            raise RuntimeError('unsupported TP4 owner geometry')
        self.owner_rows = -(-self.rows // 4)
        self.padded_rows = 4 * self.owner_rows
        # Real rows held by this rank; the tail of rank 3 (and only rank 3 once rows >= 4) is padding.
        self.valid_rows = max(0, min(self.owner_rows, self.rows - self.rank * self.owner_rows))

    def local_view(self, tensor: Any) -> Any:
        if tensor.shape[0] != self.rows:
            raise RuntimeError('mHC full-to-owner row mismatch')
        q, lo = self.owner_rows, self.rank * self.owner_rows
        if lo + q <= self.rows:
            return tensor.narrow(0, lo, q)
        out = tensor.new_empty((q, *tensor.shape[1:]))
        if self.valid_rows:
            out[:self.valid_rows] = tensor[lo:lo + self.valid_rows]
        out[self.valid_rows:] = 0
        return out

    def _check(self, tensor: Any, expected_rows: int) -> None:
        import torch
        if not self.comm.available or self.comm.disabled:
            raise RuntimeError('mHC communicator became unavailable; no local fallback')
        if (tensor.shape[0] != expected_rows or not tensor.is_cuda
                or tensor.device != self.comm.device or tensor.dtype != torch.bfloat16
                or not tensor.is_contiguous()):
            raise RuntimeError('mHC collectives require contiguous BF16 owner/full CUDA rows')

    def reduce_scatter(self, partial: Any) -> Any:
        import torch
        self._check(partial, self.rows)
        if tuple(partial.shape) != (self.rows, HIDDEN):
            raise RuntimeError('mHC requires a full-hidden TP partial')
        source = partial
        if self.padded_rows != self.rows:
            source = partial.new_empty((self.padded_rows, HIDDEN))
            source[:self.rows] = partial
            source[self.rows:] = 0
        output = partial.new_empty((self.owner_rows, HIDDEN))
        stream = torch.cuda.current_stream(partial.device)
        if self.rdma is not None:
            self.rdma.reduce_scatter(output, source)
        else:
            self.comm.reduce_scatter(output, source, stream=stream)
        partial.record_stream(stream)
        if source is not partial:
            source.record_stream(stream)
        output.record_stream(stream)
        self.rs_count += 1
        return output

    def all_gather(self, owned: Any) -> Any:
        import torch
        self._check(owned, self.owner_rows)
        output = owned.new_empty((self.padded_rows, *owned.shape[1:]))
        stream = torch.cuda.current_stream(owned.device)
        if self.rdma is not None:
            self.rdma.all_gather(output, owned)
        else:
            self.comm.all_gather(output, owned, stream=stream)
        owned.record_stream(stream)
        output.record_stream(stream)
        self.ag_count += 1
        # Leading rows of a contiguous buffer: still contiguous, and exactly the real rows.
        return output if self.padded_rows == self.rows else output[:self.rows]

    def finish(self, layers: int, auxiliary_gathers: int) -> None:
        if self.rs_count != 2 * layers or self.ag_count != 2 * layers + auxiliary_gathers:
            raise RuntimeError(f'mHC collective accounting mismatch: RS={self.rs_count} AG={self.ag_count}')
        _TOTALS['engaged'] += 1
        _TOTALS['engaged_rows'] += self.rows
        _TOTALS['padded'] += int(self.padded_rows != self.rows)
        _TOTALS['rdma'] += int(self.rdma is not None)
        key = ('engaged', self.rows)
        if _REPORTS.get(key, 0) < 2:
            _LOG.warning('GLM53_MHC_PREFILL rank=%d rows=%d owner_rows=%d padded_rows=%d rs=%d ag=%d aux=%d rdma=%d',
                         self.rank, self.rows, self.owner_rows, self.padded_rows, self.rs_count, self.ag_count,
                         auxiliary_gathers, int(self.rdma is not None))
            _REPORTS[key] = _REPORTS.get(key, 0) + 1
        if _TOTALS['engaged'] % 64 == 0:
            _LOG.warning('GLM53_MHC_PREFILL_TOTALS rank=%d engaged=%d engaged_rows=%d padded=%d '
                         'fallback_at_or_above_threshold=%d rdma=%d', self.rank, _TOTALS['engaged'],
                         _TOTALS['engaged_rows'], _TOTALS['padded'], _TOTALS['fallback_at_or_above_threshold'],
                         _TOTALS['rdma'])


def maybe_create(model: Any, hidden: Any, positions: Any) -> PrefillOwnership | None:
    if not model._mhc_prefill_enabled:
        return None
    import torch
    # No Python collectives, data-dependent branches or partial ownership in graphs.
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    from vllm.distributed import get_tp_group
    from vllm.forward_context import get_forward_context, is_forward_context_available
    if not is_forward_context_available():
        return None
    context = get_forward_context()
    if context.cudagraph_runtime_mode.name != 'NONE' or context.ubatch_slices is not None:
        return None
    group = get_tp_group()
    if group.world_size != 4:
        raise RuntimeError('mHC TP group is not four ranks')
    rows = hidden.shape[0]
    eligible, error, comm, reason = False, None, None, 'below-threshold'
    try:
        if model._mhc_prefill_min_rows <= rows <= MAX_ROWS:
            names = validate_model(model)
            reason = batch_metadata(context.attn_metadata, names, rows)
            if reason is None and not (tuple(hidden.shape) == (rows, HIDDEN) and positions.shape[0] == rows
                                       and hidden.is_cuda and hidden.dtype == torch.bfloat16
                                       and hidden.is_contiguous()):
                reason = 'hidden-geometry'
            eligible = reason is None
            if eligible:
                comm = getattr(group.device_communicator, 'pynccl_comm', None)
                if (comm is None or not comm.available or comm.disabled or comm.world_size != 4
                        or comm.rank != group.rank_in_group or comm.device != hidden.device):
                    raise RuntimeError('mHC requires the active TP PyNccl communicator on the current device')
                if torch.cuda.get_device_capability(hidden.device) != (12, 1):
                    raise RuntimeError('mHC packet is qualified for SM121 only')
    except (AttributeError, RuntimeError) as exc:
        error = str(exc)
    # Every enabled eager target forward votes, including unsupported shapes.
    # A rank cannot independently fall back after another rank elected RS/AG.
    votes = [None] * 4
    torch.distributed.all_gather_object(votes, (eligible, rows, error), group=group.cpu_group)
    if not agree_admission(votes):
        if reason != 'below-threshold':
            _TOTALS['fallback_at_or_above_threshold'] += 1
            key = ('fallback', rows)
            if _REPORTS.get(key, 0) < 2:
                _LOG.warning('GLM53_MHC_PREFILL_FALLBACK rank=%d rows=%d reason=%s',
                             group.rank_in_group, rows, reason)
                _REPORTS[key] = _REPORTS.get(key, 0) + 1
        return None
    # Every rank has the same row count here (the vote above), so all ranks pick the same transport.
    rdma = model._mhc_prefill_rdma if rows >= model._mhc_prefill_rdma_min_rows else None
    return PrefillOwnership(comm=comm, rank=group.rank_in_group, rows=rows, rdma=rdma)
