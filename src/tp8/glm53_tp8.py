# SPDX-License-Identifier: Apache-2.0
"""TP8 startup contract. Imported by the TP8 model overlay only."""
import hashlib
import ipaddress
import json
import logging
import os

ZERO_SWITCHES = (
    'B12X_TP2_CLUSTER_MAP', 'B12X_TP4_CLUSTER_MAP', 'B12X_TP2_TILE_MAJOR_WEIGHTS',
    'B12X_TP4_TILE_MAJOR_WEIGHTS', 'GLM53_TP2_MHC_GEOMETRY', 'GLM53_TP4_MHC_GEOMETRY',
    'GLM53_MHC_PREFILL_SHARD', 'GLM53_MHC_PREFILL_RDMA', 'GLM53_SPEEDUP_FAIR_COST_JOINT',
    'GLM53_RECOVERSSM_FUSED_COMMIT', 'GLM53_ADAPTIVE_FAIR_CHUNK', 'GLM53_SPEEDUP_GEOMETRY',
    'GLM53_SPEEDUP_PREFILL_ADAPTIVE', 'GLM53_SPEEDUP_SPLIT_CADENCE', 'GLM53_SPEEDUP_SPLIT_ADAPTIVE',
    'GLM53_SPEEDUP_SPLIT_FAIR', 'GLM53_SPEEDUP_MIXED_PREFILL', 'GLM53_SPEEDUP_MIXED_PREFILL_FAIR',
    'GLM53_SPEEDUP_MIXED_PREFILL_RUNTIME', 'GLM53_SPEEDUP_PREFILL_TAILS',
    'GLM53_SPEEDUP_TP2_FAIR_CAP_GEOMETRY', 'GLM53_SPEEDUP_FIRST_TOKEN_REPAY',
    'B12X_GLM53_ATOMIC_PREFILL', 'B12X_GLM53_MIXED_SPLIT', 'GLM53_INDEXER_ROW_SPLIT',
    'GLM53_SPEEDUP_ROCE', 'GLM53_SPEEDUP_MHC_SHARD', 'GLM53_SPEEDUP_KDA_STRIDES',
    'GLM53_SPEEDUP_NOPE_COPIES', 'GLM53_SPEEDUP_NOPE_LAYOUT', 'GLM53_SPEEDUP_NOPE_CONCAT',
    'B12X_GLM53_DECODE_FASTPATH', 'GLM53_INDEXER_GATE_TC', 'GLM53_SPARSE_MLA_SPLIT_KV',
    'GLM53_EMBED_REPLICATE', 'VLLM_GLM53_V2_SP_EXPERIMENTAL', 'VLLM_GLM53_V2_ASYNC_TP_EXPERIMENTAL',
)
FIXED_ENV = {
    'VLLM_GLM53_TP2_ALLREDUCE': 'nccl', 'VLLM_GLM53_TP4_LEAN_ALLREDUCE': 'nccl',
    'GLM53_SPEEDUP_CAPTURE': 'off', 'GLM53_SPEEDUP_MODE': 'off', 'GLM53_KDA_STATE': 'fp32',
    'NCCL_ALGO': 'Ring', 'NCCL_NET': 'IB', 'NCCL_SWITCHLESS_RING_ONLY': '1',
    'NCCL_IB_SUBNET_AWARE_ROUTING': '1', 'NCCL_IB_SUBNET_PREFIX_LEN': '24',
    'NCCL_IB_MERGE_NICS': '0', 'NCCL_IB_EXTENDED_IPV4_GIDS': '1',
    'NCCL_IB_PRESERVE_PCI_DOMAIN': '1',
    'VLLM_NCCL_SO_PATH': '/opt/switchless-nccl-tp8/libnccl.so.2',
}


def inspect(config, env):
    parallel = config.parallel_config
    if (parallel.tensor_parallel_size != 8 or parallel.pipeline_parallel_size != 1
            or parallel.data_parallel_size != 1 or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1 or parallel.enable_expert_parallel
            or parallel.enable_eplb or parallel.use_sequence_parallel_moe):
        raise ValueError('TP8 baseline requires TP8/PP1/DP1/DCP1/PCP1 without EP/EPLB/SP')
    for key in ZERO_SWITCHES:
        if env.get(key, '0') != '0':
            raise ValueError(f'{key} is not qualified at TP8')
    for key, expected in FIXED_ENV.items():
        if env.get(key) != expected:
            raise ValueError(f'{key} must be {expected} at TP8')
    if env.get('GLM53_SPEEDUP_OVERRIDE') or env.get('GLM53_SPEEDUP_COSTS'):
        raise ValueError('runtime policy overrides are disabled for TP8 bring-up')
    stage = env.get('GLM53_TP8_STAGE', 'target')
    if stage not in ('target', 'fixed-k', 'graphs'):
        raise ValueError('unknown TP8 stage')
    model = config.model_config
    text = getattr(model.hf_config, 'text_config', model.hf_config)
    expected = {'hidden_size': 4096, 'num_attention_heads': 64, 'index_n_heads': 32,
                'linear_num_heads': 64, 'kv_lora_rank': 512, 'q_lora_rank': 1536,
                'vocab_size': 154880}
    actual = {k: getattr(text, k, None) for k in expected}
    if actual != expected:
        raise ValueError('TP8 target configuration differs from the pinned shape contract')
    k = config.num_speculative_tokens
    if stage == 'target' and k != 0:
        raise ValueError('target stage must disable speculation')
    if stage != 'target' and k != 4:
        raise ValueError('fixed draft stages require K=4')
    if stage != 'graphs' and not model.enforce_eager:
        raise ValueError('target and fixed-k stages require eager execution')
    if stage == 'graphs' and model.enforce_eager:
        raise ValueError('graphs stage requires capture enabled')
    if config.scheduler_config.max_num_seqs > (4 if stage == 'graphs' else 1):
        raise ValueError('bring-up concurrency exceeds the stage contract')
    if not 0 < config.cache_config.kv_cache_memory_bytes <= 8 * 1024**3:
        raise ValueError('bring-up requires an explicit cache budget of at most 8 GiB per rank')
    if model.max_model_len > 8192:
        raise ValueError('bring-up is limited to 8192 tokens; capacity needs a separate qualification')
    cables = env.get('NCCL_SWITCHLESS_CABLE_SUBNETS', '')
    nets = [ipaddress.ip_network(value, strict=True) for value in cables.split(',')]
    if len(nets) != 8 or len(set(nets)) != 8 or any(n.version != 4 or n.prefixlen != 24 for n in nets):
        raise ValueError('eight distinct IPv4 /24 cable subnets required')
    return {'stage': stage, 'target': actual, 'local_mla_heads': 8, 'local_kda_heads': 8,
            'local_indexer_heads': 32, 'residual_hidden': 4096, 'speculative_tokens': k,
            'max_model_len': model.max_model_len, 'max_num_seqs': config.scheduler_config.max_num_seqs,
            'kv_cache_memory_bytes': config.cache_config.kv_cache_memory_bytes,
            'cable_subnets': cables, 'status': 'GPU_UNQUALIFIED'}


def configure(config):
    import torch
    from vllm.distributed import get_tp_group
    group = get_tp_group()
    record, error = None, None
    try:
        record = inspect(config, os.environ)
        if group.world_size != 8:
            raise ValueError('actual TP communicator does not have eight ranks')
    except (ValueError, AttributeError, TypeError) as exc:
        error = str(exc)
    fingerprint = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    votes = [None] * group.world_size
    torch.distributed.all_gather_object(votes, (fingerprint, error), group=group.cpu_group)
    if any(v != (fingerprint, None) for v in votes):
        raise RuntimeError('TP8 startup contract failed or differs across ranks: ' + repr(votes))
    logging.getLogger('vllm.glm53_tp8').warning('GLM53_TP8_CONTRACT rank=%d digest=%s record=%s',
                                              group.rank_in_group, fingerprint, json.dumps(record, sort_keys=True))
