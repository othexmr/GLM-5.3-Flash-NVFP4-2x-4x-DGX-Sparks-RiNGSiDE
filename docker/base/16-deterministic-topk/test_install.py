"""Verify exact reconstruction and reject an unexpected source before editing."""
import hashlib,json,tempfile
from pathlib import Path
from install import D,install

manifest=json.loads((D/'source-manifest.json').read_text())
rel='vllm/model_executor/layers/sparse_attn_indexer_kpool.py'
original=(D/'original/sparse_attn_indexer_kpool.py').read_bytes()
assert hashlib.sha256(original).hexdigest()==manifest['files'][rel]['before']
with tempfile.TemporaryDirectory()as td:
    root=Path(td);target=root/rel;target.parent.mkdir(parents=True)
    target.write_bytes(original+b'\n# unexpected input\n')
    before=target.read_bytes()
    try:install(root)
    except RuntimeError:pass
    else:raise AssertionError('Modified source was accepted')
    assert target.read_bytes()==before and not (root/'deterministic_sparse_topk.py').exists()
    target.write_bytes(original);install(root)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==manifest['files'][rel]['after']
print('Deterministic top-k exact reconstruction and preimage rejection PASS')
