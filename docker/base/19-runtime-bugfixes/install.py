from pathlib import Path
import hashlib,json,sysconfig,py_compile,shutil
root=Path(__file__).resolve().parent
site=Path(sysconfig.get_paths()['purelib'])
manifest=json.loads((root/'source-manifest.json').read_text())
for item in manifest:
    dst=site/item['path'];src=root/'runtime-candidate'/item['path']
    current=hashlib.sha256(dst.read_bytes()).hexdigest() if dst.exists() else None
    if current != item['before']: raise RuntimeError(f"base source drift: {dst}")
    if hashlib.sha256(src.read_bytes()).hexdigest()!=item['after']:raise RuntimeError(f"candidate source drift: {src}")
for item in manifest:
    dst=site/item['path'];dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(root/'runtime-candidate'/item['path'],dst)
    py_compile.compile(str(dst),doraise=True)
print('Installed hash-verified runtime safety fixes')
