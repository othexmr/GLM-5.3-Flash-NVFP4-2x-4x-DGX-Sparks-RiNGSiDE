"""Apply the exact deterministic top-k callsites and install their helper."""
import hashlib,json,py_compile,shutil,subprocess
from pathlib import Path

D=Path(__file__).resolve().parent

def install(root):
    manifest=json.loads((D/'source-manifest.json').read_text())
    for rel,row in manifest['files'].items():
        target=root/rel
        if row['before'] is None:
            if target.exists():raise RuntimeError('Unexpected existing payload: '+rel)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=row['before']:
            raise RuntimeError('Unexpected source preimage: '+rel)
    patch=D/'callsite.patch'
    if hashlib.sha256(patch.read_bytes()).hexdigest()!=manifest['patch_sha256']:
        raise RuntimeError('Patch hash differs')
    helper=D/'deterministic_sparse_topk.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=manifest['files'][helper.name]['after']:
        raise RuntimeError('Helper hash differs')
    subprocess.run(['patch','--batch','--fuzz=0','-p1','-i',str(patch)],cwd=root,check=True)
    shutil.copyfile(helper,root/helper.name)
    for rel,row in manifest['files'].items():
        target=root/rel
        if hashlib.sha256(target.read_bytes()).hexdigest()!=row['after']:
            raise RuntimeError('Installed payload differs: '+rel)
        py_compile.compile(str(target),doraise=True)
    return manifest

if __name__=='__main__':
    install(Path('/usr/local/lib/python3.12/dist-packages'))
    print('Deterministic sparse top-k payloads installed and verified')
