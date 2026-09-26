#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Image-local preflight. Read only: explicit model files, installed consumers, sysfs and memory."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'fabric'))
import preflight


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def memory():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:')))*1024


def check(config):
    from contracts import digest, RESERVE_BYTES
    report = preflight.check(Path('/sys/class/infiniband'),config['topology'],config['rank'],3)
    if not report['ok']: raise ValueError('local fabric does not match declared topology')
    assets=json.loads((ROOT/'release/assets.json').read_text())
    target=Path('/models/target').resolve()
    index=target/'model.safetensors.index.json'
    model_asset=next(m for m in assets['models'] if m['name']=='RedHatAI/GLM-5.3-Flash-NVFP4')
    if file_hash(index)!=model_asset['index_sha256']: raise ValueError('target index differs from pinned revision')
    tokenizer=next(m for m in assets['models'] if m.get('file')=='tokenizer.json')
    token_hash=file_hash(target/'tokenizer.json')
    if token_hash!=tokenizer['sha256']: raise ValueError('tokenizer override missing or wrong')
    filenames=set(json.loads(index.read_text())['weight_map'].values())|{'config.json','model.safetensors.index.json'}
    hashes={}
    for name in sorted(filenames):
        path=(target/name).resolve()
        cached = target.parent.name=='snapshots' and path.is_relative_to(target.parent.parent/'blobs')
        if Path(name).name!=name or not (path.is_relative_to(target) or cached) or not path.is_file():
            raise ValueError('missing or unsafe target shard')
        hashes[name]=dict(sha256=file_hash(path),bytes=path.stat().st_size)
    spec=importlib.util.spec_from_file_location('served',ROOT/'docker/tools/check_served.py')
    served=importlib.util.module_from_spec(spec);spec.loader.exec_module(served)
    installed=served.check('tp8',ROOT/'sources/installed-files.json',Path('/'),config['rank'])
    if any(r['status'] in ('missing','different') for r in installed): raise ValueError('installed consumer mismatch')
    mem=memory()
    return dict(rank=config['rank'],ok=mem>=RESERVE_BYTES, image_id=config['image_id'],
        profile_sha256=file_hash(ROOT/'launch/profiles/tp8/profile.json'),installed_sha256=digest(installed),
        target_sha256=digest(hashes),tokenizer_sha256=token_hash,topology_sha256=report['topology_sha256'],
        nccl_sha256=file_hash('/opt/switchless-nccl-tp8/libnccl.so.2'),kit_sha256=config['kit_sha256'],
        launch_config_sha256=config['launch_config_sha256'],
        mem_available_bytes=mem,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        fabric=report,shards=hashes,installed=installed)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    config=json.loads(a.input.read_text()); result=dict(rank=config['rank'],ok=False)
    try: result=check(config)
    except BaseException as exc:
        result.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:a.out.write_text(json.dumps(result,indent=1)+'\n')
