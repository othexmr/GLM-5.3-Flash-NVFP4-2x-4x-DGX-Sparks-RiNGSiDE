# SPDX-License-Identifier: Apache-2.0
"""Read only the pinned DFlash2 snapshot selected by the stage profile; no Hugging Face credentials."""
import hashlib
import json
from pathlib import Path
import sys

REVISION='7d74cdd881ed7e32c31175984a67823127b66cfe'
EXPECTED=dict(hidden_size=4096,num_attention_heads=32,num_key_value_heads=8,head_dim=128,
              num_hidden_layers=5,vocab_size=154880)


def check(root):
    root=(Path(root)/'snapshots'/REVISION).resolve()
    config=json.loads((root/'config.json').read_text())
    if {k:config.get(k) for k in EXPECTED}!=EXPECTED:raise ValueError('drafter geometry differs')
    if config.get('dflash_config',{}).get('target_layer_ids')!=[5,14,24,33,42]:raise ValueError('drafter capture layers differ')
    index=root/'model.safetensors.index.json'
    names=set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    files={}
    for name in sorted(names|{'config.json'}):
        path=(root/name).resolve()
        # HF cache snapshots commonly use symlinks into the same cache repository's blobs directory.
        if (Path(name).name!=name or not path.is_file()
                or not (path.is_relative_to(root) or path.is_relative_to(root.parent.parent/'blobs'))):
            raise ValueError('missing or unsafe drafter shard name')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        files[name]=h.hexdigest()
    return dict(revision=REVISION,heads_per_rank=4,kv_heads_per_rank=1,
                sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),files=files)


if __name__=='__main__':print(json.dumps(check(sys.argv[1])))
