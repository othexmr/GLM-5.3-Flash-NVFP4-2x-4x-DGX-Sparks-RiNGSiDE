# SPDX-License-Identifier: Apache-2.0
"""Pure receipt, route and stage contracts; no hardware access."""
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import time

STAGES = ('transport', 'target', 'fixed-k', 'graphs')
RESERVE_BYTES = 16 * 1024**3
# A continuous load interval includes packing, warm-up and (for the graph stage) capture.
# It does not claim separate phase peaks without matching engine instrumentation.
LIFECYCLE = ('before-load', 'load', 'ready', 'serve')
HASH = re.compile(r'[0-9a-f]{64}')
IDENTITY_FIELDS = ('image_id', 'profile_sha256', 'installed_sha256', 'target_sha256',
                   'tokenizer_sha256', 'topology_sha256', 'nccl_sha256', 'kit_sha256',
                   'launch_config_sha256')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def identity(rows, ranks):
    require(len(rows) == ranks and {x['rank'] for x in rows} == set(range(ranks)), 'missing or duplicate ranks')
    require(all(x['ok'] is True for x in rows), 'a rank failed preflight')
    fields = IDENTITY_FIELDS
    for field in fields:
        require(len({x[field] for x in rows}) == 1, field + ' differs across ranks')
        value = rows[0][field]
        require(HASH.fullmatch(value.removeprefix('sha256:')) is not None, 'invalid identity: ' + field)
    require(all(x['mem_available_bytes'] >= RESERVE_BYTES for x in rows), 'less than 16 GiB memory reserve')
    return {field: rows[0][field] for field in fields}


def admit(receipt, requested, expected, now=None):
    require(requested in STAGES[1:], 'unknown serving stage')
    prior = STAGES[STAGES.index(requested)-1]
    require(receipt.get('schema') == 'tp8-test/1' and receipt.get('ok') is True, 'a passing kit receipt is required')
    require(receipt.get('stage') == prior and receipt.get('ranks') == 8, 'previous eight-rank stage is required')
    require(receipt.get('identity') == expected, 'receipt does not match this candidate')
    require(receipt.get('failures') == [], 'receipt contains failures')
    age = (time.time() if now is None else now) - receipt['finished_unix']
    require(0 <= age <= 24*3600, 'receipt is stale or from the future')
    require(receipt.get('source_receipts_verified') is True, 'raw rank receipts were not checked')
    if prior != 'transport':
        require(receipt.get('completion_gate') is True, 'completed generation and quality checks required')
        require(receipt.get('memory_gate') is True, 'full memory lifecycle gate required')
    if prior == 'fixed-k':
        require(receipt.get('speculation_gate') is True, 'acceptance, rejection and prompt replay checks required')
    return True


def memory_gate(samples, stage, ranks=8):
    required = set(LIFECYCLE)
    for rank in range(ranks):
        rows = [r for r in samples if r['rank'] == rank]
        require(required <= {r['phase'] for r in rows}, f'rank {rank}: incomplete memory lifecycle')
        require(all(type(r['mem_available_bytes']) is int and r['mem_available_bytes'] >= RESERVE_BYTES for r in rows),
                f'rank {rank}: memory reserve breached')
        require(len(rows)>4 and all(0 < b['monotonic']-a['monotonic'] <= 1.0 for a,b in zip(rows,rows[1:])),
                f'rank {rank}: incomplete continuous memory sampling')
    return True


def route_evidence(text, topology, rank):
    """Require observed NCCL channels and final IB endpoints; env selection alone is insufficient."""
    adjacent, local, owner = set(), {}, {}
    for edge in topology['links']:
        for r in edge['ranks']:
            for hca, address in edge['addresses'][str(r)].items():
                owner[address] = (r, hca)
        if rank in edge['ranks']:
            adjacent.update(set(edge['ranks']) - {rank})
            local.update(edge['addresses'][str(rank)])
    rings = re.findall(r'Ring\s+(\d+)\s*:\s*(\d+)\s*->\s*(\d+)\s*->\s*(\d+)', text)
    require(bool(rings), 'no actual NCCL ring records')
    for channel, prev, current, nxt in rings:
        require(int(current) == rank and {int(prev), int(nxt)} == adjacent, 'NCCL selected a non-cable ring edge')
    channels = re.findall(r'Channel\s+(\d+)/\d+\s*:\s*(\d+)\[\d+\]\s*->\s*(\d+)\[\d+\].*?via NET/([^\s]+)', text)
    require(bool(channels), 'no actual network channel records')
    for channel, src, dst, transport in channels:
        require(transport.startswith('IB/') and rank in (int(src),int(dst)), 'unexpected channel transport or rank')
        require(({int(src),int(dst)} - {rank}) <= adjacent, 'channel reaches a non-neighbor')
    routes = []
    for line in text.splitlines():
        if 'NET/IB RouteFinal ' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        a = str(ipaddress.IPv6Address(int(fields['localGid'], 16)).ipv4_mapped)
        b = str(ipaddress.IPv6Address(int(fields['remoteGid'], 16)).ipv4_mapped)
        hca = fields['hca']
        require(local.get(hca) == a and b in owner, 'undeclared RDMA endpoint')
        peer, peer_hca = owner[b]
        require(peer in adjacent and peer_hca == hca, 'RDMA peer or PCI root differs from the cable contract')
        require(ipaddress.ip_network(a+'/24',False) == ipaddress.ip_network(b+'/24',False), 'RDMA crossed subnets')
        routes.append(dict(side=fields['side'], hca=hca, local=a, remote=b, peer_rank=peer))
    require(bool(routes) and {r['side'] for r in routes} == {'send','recv'}, 'missing final RDMA route records')
    require({r['peer_rank'] for r in routes} == adjacent, 'both cable peers were not observed')
    require({r['hca'].startswith('roceP2') for r in routes} == {False, True}, 'both PCI roots were not observed')
    tuning = re.findall(r'(AllReduce|AllGather): (\d+) Bytes -> Algo (\w+) proto (\w+) channel\{Lo\.\.Hi\}=\{(\d+)\.\.(\d+)\}', text, re.I)
    if rank == 0:
        require({t[0].lower() for t in tuning} == {'allreduce','allgather'}, 'rank zero lacks actual algorithm records')
        require(all(t[2].lower() == 'ring' for t in tuning), 'a tested collective selected another algorithm')
    return dict(rings=[list(map(int,r)) for r in rings], routes=routes, network='IB', tuning=tuning)


def bundle(receipt):
    """Allowlist typed results. No environment, headers, raw logs, output text or arbitrary strings."""
    out = {k: receipt[k] for k in ('schema','stage','ranks','ok','finished_unix') if k in receipt}
    require(out.get('schema') == 'tp8-test/1' and out.get('stage') in STAGES, 'unsupported bundle schema')
    require(type(out.get('ranks')) is int and out['ranks'] in (4,8) and type(out.get('ok')) is bool,
            'invalid bundle scalar fields')
    require(type(out.get('finished_unix')) in (int,float) and math.isfinite(out['finished_unix']), 'invalid timestamp')
    out['identity'] = {}
    for key in IDENTITY_FIELDS:
        if key in receipt.get('identity', {}):
            value = receipt['identity'][key]
            require(type(value) is str and HASH.fullmatch(value.removeprefix('sha256:')), 'bad hash')
            out['identity'][key] = value
    out['metrics'] = []
    for row in receipt.get('metrics', []):
        clean = {k: v for k,v in row.items() if k in (
            'rank','tokens','prompt_tokens','completion_tokens','ttft_ms','elapsed_ms','decode_tps',
            'bytes','median_ms','context','concurrency','repetition','mem_available_bytes')
            and type(v) in (int,float) and math.isfinite(v)}
        for key,allowed in {
            'operation':('allreduce','allgather'), 'dtype':('torch.int32','torch.float32'),
            'case':('cold','replay','concurrent','long-first','short-first'),
        }.items():
            if row.get(key) in allowed:clean[key]=row[key]
        out['metrics'].append(clean)
    out['gates']={k:receipt[k] for k in ('completion_gate','memory_gate','speculation_gate')
                  if type(receipt.get(k)) is bool}
    out['memory_min_bytes']={k:v for k,v in receipt.get('memory_min_bytes',{}).items()
                             if k in tuple(map(str,range(8))) and type(v) is int and v>=0}
    out['speculation_counts']={k:v for k,v in receipt.get('speculation_counts',{}).items()
                               if k in ('draft','accepted','rejected') and type(v) in (int,float) and math.isfinite(v)}
    out['failure_phases']=[r['phase'] for r in receipt.get('failures',[])
                           if isinstance(r,dict) and r.get('phase') in ('node','collective','stage','memory','cleanup')]
    out['routes'] = []
    pseudonyms = {}
    def pseudo(address):
        # Validate the address, then replace it consistently within this bundle.
        ipaddress.ip_address(address)
        if address not in pseudonyms: pseudonyms[address] = 'endpoint-'+str(len(pseudonyms))
        return pseudonyms[address]
    for row in receipt.get('routes', []):
        require(type(row['rank']) is int and type(row['peer_rank']) is int, 'invalid route rank')
        out['routes'].append(dict(rank=row['rank'], peer_rank=row['peer_rank'],
                                 local=pseudo(row['local']), remote=pseudo(row['remote'])))
    out['gpu_qualification'] = 'not-established'  # bring-up is separate from performance qualification
    return out
