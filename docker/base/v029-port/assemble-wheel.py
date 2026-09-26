import base64,csv,hashlib,io,json,pathlib,re,shutil,zipfile
root=pathlib.Path('/assembly');root.mkdir(exist_ok=True)
wheels=list(pathlib.Path('/native-wheels').glob('*.whl'));assert len(wheels)==1,wheels
with zipfile.ZipFile(wheels[0]) as z:z.extractall(root)
shutil.copytree('/ported-source/vllm',root/'vllm',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
for p in pathlib.Path('/rust-artifacts').iterdir():
 if p.name.startswith('_rust_') and p.suffix=='.so' or p.name=='vllm-rs':shutil.copy2(p,root/'vllm'/p.name)
assert list((root/'vllm').glob('_rust_*.so'))
assert (root/'vllm'/'vllm-rs').is_file()
version='0.29.0+glm53.local'
(root/'vllm'/'_version.py').write_text(f'__version__ = {version!r}\n__version_tuple__ = (0, 29, 0, "glm53.local")\n')
infos=list(root.glob('*.dist-info'));assert len(infos)==1
info=root/f'vllm-{version}.dist-info';infos[0].rename(info)
meta=info/'METADATA';meta.write_text(re.sub(r'^Version: .*$',f'Version: {version}',meta.read_text(),flags=re.M))
record=info/'RECORD';record.unlink(missing_ok=True)
rows=[];manifest={}
for p in sorted(root.rglob('*')):
 if p.is_file():
  rel=p.relative_to(root).as_posix();b=p.read_bytes();h=hashlib.sha256(b).digest();rows.append([rel,'sha256='+base64.urlsafe_b64encode(h).decode().rstrip('='),str(len(b))]);manifest[rel]=h.hex()
rows.append([record.relative_to(root).as_posix(),'',''])
with record.open('w',newline='') as f:csv.writer(f).writerows(rows)
out=pathlib.Path('/assembled-wheels');out.mkdir(exist_ok=True)
tags=wheels[0].name.split('-')[-3:];dest=out/('-'.join(['vllm',version,*tags]))
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in sorted(root.rglob('*')):
  if p.is_file():z.write(p,p.relative_to(root))
(out/'manifest.json').write_text(json.dumps({'version':version,'native_wheel':wheels[0].name,'files':manifest},indent=2)+'\n')
print(dest)
