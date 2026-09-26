#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run in one owned container per rank, before model load. Requires the image's PyTorch/vLLM."""
import argparse
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import statistics
import threading
import time


def peers(topology, rank, port, nonce):
    """Bounded bidirectional nonce exchange, bound to each declared local fabric address."""
    pairs = []
    for edge in topology['links']:
        if rank not in edge['ranks']: continue
        peer = next(r for r in edge['ranks'] if r != rank)
        for hca, local in edge['addresses'][str(rank)].items():
            pairs.append((local, edge['addresses'][str(peer)][hca]))
    sockets, result = [], []
    try:
        for local, remote in pairs:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind((local,port)); s.settimeout(.2); sockets.append((s,local,remote))
        deadline = time.monotonic()+20
        seen = set()
        while time.monotonic() < deadline:
            for s,local,remote in sockets:
                s.sendto(nonce.encode(), (remote,port))
                try:
                    data, sender = s.recvfrom(256)
                    if data == nonce.encode() and sender == (remote,port): seen.add(local)
                except socket.timeout:
                    pass
        if len(seen) == len(sockets):
            return [{'local':local,'remote':remote,'ok':True} for _,local,remote in sockets]
        raise RuntimeError('fabric nonce exchange did not complete on every endpoint')
    finally:
        for s,_,_ in sockets: s.close()


def run(config):
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    rank, n = config['rank'], len(config['topology']['nodes'])
    if n not in (4,8): raise ValueError('four/eight-rank leaf only')
    library = Path(config['library']).resolve()
    sha = hashlib.sha256(library.read_bytes()).hexdigest()
    if sha != config['nccl_sha256']: raise ValueError('native library differs from receipt')
    torch.cuda.set_device(0)
    dist.init_process_group('gloo', init_method=config['rendezvous'], rank=rank, world_size=n,
                            timeout=datetime.timedelta(seconds=90))
    result = dict(rank=rank, ok=False, nccl_sha256=sha, metrics=[], peers=[], world_size=n)
    try:
        votes = [None]*n
        fingerprint = hashlib.sha256(json.dumps({k:config[k] for k in ('topology','nccl_sha256','identity')},sort_keys=True).encode()).hexdigest()
        dist.all_gather_object(votes, fingerprint)
        if len(set(votes)) != 1: raise ValueError('rank configuration identities differ')
        result['peers'] = peers(config['topology'], rank, config['peer_port'], config['nonce'])
        dist.barrier()
        comm = PyNcclCommunicator(dist.group.WORLD, device=0, library_path=str(library))
        if not comm.available or comm.disabled: raise RuntimeError('exact-library NCCL communicator unavailable')
        maps = Path('/proc/self/maps').read_text()
        if str(library) not in maps: raise RuntimeError('requested library absent from loaded process maps')
        result['nccl_version'] = comm.nccl_version
        stream = torch.cuda.current_stream()
        # Cover scalar/control messages and multi-megabyte prefill transfers. Integer values sum exactly in fp32.
        for count in (1,32,1024,16384,262144,2097152):
            for dtype in (torch.int32,torch.float32):
                source = (torch.arange(count,device='cuda')%31 + rank*37).to(dtype)
                expected = (torch.arange(count,device='cuda')%31*n + 37*n*(n-1)//2).to(dtype)
                reduced = torch.empty_like(source)
                gathered = torch.empty(n*count,device='cuda',dtype=dtype)
                for operation in ('allreduce','allgather'):
                    samples=[]
                    for repetition in range(7):
                        dist.barrier(); torch.cuda.synchronize()
                        begin=time.perf_counter_ns()
                        if operation=='allreduce': comm.all_reduce(source, out_tensor=reduced,stream=stream)
                        else: comm.all_gather(gathered,source,stream=stream)
                        torch.cuda.synchronize(); samples.append((time.perf_counter_ns()-begin)/1e6)
                        if operation=='allreduce':
                            if not torch.equal(reduced,expected): raise ValueError('all-reduce numerical mismatch')
                        else:
                            for peer in range(n):
                                want=(torch.arange(count,device='cuda')%31+peer*37).to(dtype)
                                if not torch.equal(gathered[peer*count:(peer+1)*count],want):
                                    raise ValueError('all-gather rank order or data mismatch')
                    result['metrics'].append(dict(rank=rank, operation=operation, dtype=str(dtype),
                        bytes=count*source.element_size(), median_ms=statistics.median(samples[2:]), samples_ms=samples))
        result.update(ok=True, status='COLLECTIVE_PASS_SERVING_UNQUALIFIED')
    finally:
        dist.destroy_process_group()
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    config=json.loads(a.input.read_text()); result=dict(rank=config['rank'],ok=False)
    try:
        result=run(config)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        a.out.write_text(json.dumps(result,indent=1)+'\n')
