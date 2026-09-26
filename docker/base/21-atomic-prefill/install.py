import hashlib,json,py_compile,sys,sysconfig
from pathlib import Path
if sys.flags.optimize:raise RuntimeError("Optimized Python is forbidden for source installation")
D=Path(__file__).parent;site=Path(sysconfig.get_paths()['purelib']);rows=json.loads((D/'source-manifest.json').read_text())
for r in rows:
 p=site/r['path'];src=D/'source'/r['path']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=r['before']:raise RuntimeError('Installed preimage mismatch: '+r['path'])
 if hashlib.sha256(src.read_bytes()).hexdigest()!=r['after']:raise RuntimeError('Candidate source mismatch: '+r['path'])
for r in rows:
 p=site/r['path'];p.write_bytes((D/'source'/r['path']).read_bytes());py_compile.compile(str(p),doraise=True)
print('SPEED_CANDIDATE_SOURCE_INSTALL_PASS',len(rows))
