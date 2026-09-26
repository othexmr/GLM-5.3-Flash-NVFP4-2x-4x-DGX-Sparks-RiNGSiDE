#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Experimental TP8 tester kit. Remote actions require --go; defaults only validate and print a plan."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import io
import ipaddress
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'fabric'))
import ring
from contracts import admit, bundle, digest, identity, require, route_evidence, memory_gate


def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


compose=module('compose','launch/compose.py')


def read(path): return json.loads(Path(path).read_text())
def write(path,value): Path(path).write_text(json.dumps(value,indent=1)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ssh(host,argv,*,input=None,timeout=900,check=True):
    require(not host.startswith('-') and '\n' not in host, 'unsafe SSH destination')
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,shlex.join(argv)],
                          input=input,capture_output=True,timeout=timeout,check=check)


def put(host,path,data):
    code='import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); p.open("xb").write(sys.stdin.buffer.read())'
    ssh(host,['python3','-c',code,path],input=data)


def context(site_path,topology_path):
    site=compose.render.read_site(site_path);topology=read(topology_path);ring.validate(topology)
    require(len(topology['nodes'])==8,'TP8 tester kit requires eight ranks')
    profile=compose.load_profile('tp8')
    hosts=[]
    for r in range(8):
        ips=site.get('SWITCHLESS_SSH_TARGETS',site.get('SWITCHLESS_RANK_HOST_IPS','')).split(',')
        require(len(ips)==8,'eight SSH destinations required')
        hosts.append(site.get(f'SWITCHLESS_RANK{r}_SSH') or ips[r].strip())
        values=compose.render.rank_values(site,str(r),8)
        require(values['SWITCHLESS_RANK_HOST_IP']==next(n['management_address'] for n in topology['nodes'] if n['rank']==r),
                'site rank address differs from topology')
        compose.baked_argv(profile,site,r)
    require(site.get('SWITCHLESS_CABLE_SUBNETS')==ring.cable_subnets(topology),'site cable list differs from topology')
    cidr=ipaddress.ip_network(site['SWITCHLESS_FABRIC_CIDR'])
    require(all(ipaddress.ip_network(e['subnet']).subnet_of(cidr) for e in topology['links']), 'fabric filter excludes a cable')
    require(1024<=int(site['SWITCHLESS_MASTER_PORT'])<=65533, 'master port must leave room for two test ports')
    return site,topology,profile,hosts


def packet():
    paths=['fabric/preflight.py','fabric/preflight_tp4.py','fabric/ring.py','release/assets.json',
           'sources/installed-files.json','docker/tools/check_served.py','launch/profiles/tp8/profile.json']
    paths += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'tools/tp8-test').glob('*.py'))]
    files={p:(ROOT/p).read_bytes() for p in paths}
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as t:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);entry.mode=0o644;t.addfile(entry,io.BytesIO(data))
    return buf.getvalue(),digest({k:hashlib.sha256(v).hexdigest() for k,v in files.items()})


def launch_config(site, profile, hosts):
    # Hash site data without copying it into a shareable receipt. Bind mount paths,
    # rank destinations, runtime flags and the image selection, including options
    # used by later stages. The installed image ID is checked separately.
    return digest(dict(site=site,hosts=hosts,
                       argv=[compose.baked_argv(profile,site,r) for r in range(8)]))


def verified_receipt(path):
    rec=read(path);base=Path(path).resolve().parent
    require(bool(rec.get('source_receipts')), 'receipt has no raw evidence')
    for entry in rec['source_receipts']:
        file=(base/entry['file']).resolve()
        require(file.is_relative_to(base) and file.is_file() and sha(file)==entry['sha256'], 'raw evidence missing or changed')
    return rec


def transport(a):
    site,topo,profile,hosts=context(a.site,a.topology)
    image=compose.build_image(site,profile)
    if not a.go:
        print(json.dumps(dict(action='transport',ranks=8,image=image,host_count=len(hosts),
                              checks=['installed bytes','checkpoint and tokenizer','fabric endpoints','allreduce','allgather']),indent=1));return
    out=a.out.resolve();out.mkdir(mode=0o700,parents=True,exist_ok=False)
    nonce=uuid.uuid4().hex;remote='/tmp/ringside-tp8-'+nonce
    archive,kit_hash=packet();names={r:'tp8-test-'+nonce[:12]+'-'+str(r) for r in range(8)}
    configs={};record=dict(schema='tp8-test/1',stage='transport',ranks=8,ok=False,failures=[],
                           source_receipts=[],source_receipts_verified=False,metrics=[],routes=[])
    def docker(rank,entry,cap):
        c=configs[rank]
        argv=['timeout','--kill-after=15',str(cap),'docker','run','--rm','--pull=never','--network','host',
              '--ipc','host','--gpus','all','--device','/dev/infiniband','--ulimit','memlock=-1:-1',
              '--name',names[rank],'--label','ringside.tp8-test='+nonce,
              '--mount','type=bind,src='+remote+',dst=/kit',
              '--mount','type=bind,src='+site['SWITCHLESS_MODEL_DIR']+',dst=/models/target,readonly',
              '--mount','type=bind,src='+site['SWITCHLESS_TOKENIZER_FILE']+',dst=/models/target/tokenizer.json,readonly']
        rendered=compose.baked_argv(profile,site,rank)
        env={x.split('=',1)[0]:x.split('=',1)[1] for i,x in enumerate(rendered) if i and rendered[i-1]=='-e'}
        for key,value in env.items():
            if key.startswith('NCCL_') or key in ('GLOO_SOCKET_IFNAME','VLLM_NCCL_SO_PATH'):
                argv+=['-e',key+'='+value]
        argv+=['-e','NCCL_DEBUG_FILE=/kit/'+entry+'-nccl.log','--entrypoint','python3',c['image_id'],
               '-B','/kit/tools/tp8-test/'+entry+'.py','--input','/kit/config.json','--out','/kit/'+entry+'.json']
        try:
            p=ssh(hosts[rank],argv,timeout=cap+60,check=False)
            (out/f'rank{rank}-{entry}.log').write_bytes(p.stdout+p.stderr)
            data=ssh(hosts[rank],['cat',remote+'/'+entry+'.json']).stdout
            (out/f'rank{rank}-{entry}.json').write_bytes(data)
            require(p.returncode==0,'rank failed '+entry)
            return json.loads(data)
        except BaseException:
            record['failures'].append(dict(rank=rank,phase=entry));raise
    def initialize(rank):
        ident=ssh(hosts[rank],['docker','image','inspect','--format','{{.Id}}',image],timeout=30).stdout.decode().strip()
        configs[rank]=dict(rank=rank,topology=topo,image_id=ident,kit_sha256=kit_hash,
                           launch_config_sha256=launch_config(site,profile,hosts))
        put(hosts[rank],remote+'/packet.tar',archive)
        ssh(hosts[rank],['tar','-xf',remote+'/packet.tar','-C',remote],timeout=60)
        put(hosts[rank],remote+'/config.json',json.dumps(configs[rank]).encode())
        return docker(rank,'node',3600)
    def collective(rank):
        c=configs[rank]
        c.update(identity=record['identity'],nccl_sha256=record['identity']['nccl_sha256'],
                 library='/opt/switchless-nccl-tp8/libnccl.so.2',nonce=nonce,
                 rendezvous='tcp://'+site['SWITCHLESS_MASTER_ADDR']+':'+str(int(site['SWITCHLESS_MASTER_PORT'])+1),
                 peer_port=int(site['SWITCHLESS_MASTER_PORT'])+2)
        # Replace only this run's config, after the first container finished.
        ssh(hosts[rank],['rm',remote+'/config.json'])
        put(hosts[rank],remote+'/config.json',json.dumps(c).encode())
        row=docker(rank,'collective',300)
        log=ssh(hosts[rank],['cat',remote+'/collective-nccl.log']).stdout
        (out/f'rank{rank}-nccl.log').write_bytes(log)
        routes=route_evidence(log.decode(),topo,rank)
        row['route_evidence']=routes
        write(out/f'rank{rank}-collective.json',row)
        return row
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            rows=list(pool.map(initialize,range(8)))
            record['identity']=identity(rows,8)
            record['rank_preflight']=rows
            results=list(pool.map(collective,range(8)))
        require(all(r['ok'] for r in results),'collective correctness failed')
        record['metrics']=[m for r in results for m in r['metrics']]
        record['routes']=[dict(route,rank=r['rank']) for r in results for route in r['route_evidence']['routes']]
        record.update(ok=True,source_receipts_verified=True)
    finally:
        for rank,host in enumerate(hosts):
            try:
                label=ssh(host,['docker','inspect','--format','{{index .Config.Labels "ringside.tp8-test"}}',names[rank]],check=False,timeout=20)
                if label.returncode==0 and label.stdout.decode().strip()==nonce:
                    ssh(host,['docker','rm','-f',names[rank]],timeout=30)
            except BaseException:
                record['failures'].append(dict(rank=rank,phase='cleanup'));record['ok']=False
        for file in sorted(out.iterdir()):
            if file.is_file(): record['source_receipts'].append(dict(file=file.name,sha256=sha(file)))
        record['finished_unix']=time.time();write(out/'receipt.json',record)
    require(record['ok'],'transport failed; preserve the output directory')
    print(out/'receipt.json')


def validate_launch(a):
    site,topo,profile,hosts=context(a.site,a.topology)
    previous=verified_receipt(a.receipt)
    expected=dict(previous['identity'],topology_sha256=ring.digest(topo),
                  profile_sha256=sha(ROOT/'launch/profiles/tp8/profile.json'),kit_sha256=packet()[1],
                  launch_config_sha256=launch_config(site,profile,hosts))
    admit(previous,a.stage,expected)
    if a.stage != 'target':
        variant = read(ROOT/'launch/variants/tp8'/ (a.stage+'.json'))
        importer = module('importer','sources/import_plan.py')
        require(variant==importer.tp8_variants(ROOT)[a.stage], 'stage variant differs from importer derivation')
        profile = variant['profile']
    return site,topo,profile,hosts,previous


def cleanup_owned(launched):
    failures=[]
    for host,container_id in reversed(launched):
        # IDs come only from this invocation's successful docker run outputs.
        # A timed-out graceful stop must not prevent cleanup of other ranks.
        try:ssh(host,['docker','stop','--time','15',container_id],check=False,timeout=30)
        except (subprocess.SubprocessError,OSError):pass
        try:ssh(host,['docker','rm','-f',container_id],timeout=30)
        except (subprocess.SubprocessError,OSError):failures.append(dict(phase='cleanup'))
    return failures


def launch(a):
    site,topo,profile,hosts,previous=validate_launch(a)
    for r in range(8):
        argv=compose.baked_argv(profile,site,r,image=previous['identity']['image_id'])
        if not a.go: print(shlex.join(argv))
    if not a.go:return
    if a.stage!='target':
        code=(ROOT/'tools/tp8-test/drafter.py').read_text()
        draft=[]
        for host in hosts:
            draft.append(json.loads(ssh(host,['python3','-c',code,site['SWITCHLESS_DRAFTER_REPO_DIR']]).stdout))
        require(len({r['sha256'] for r in draft})==1,'drafter checkpoint differs across ranks')
        if a.stage=='graphs':require(previous.get('drafter_sha256')==draft[0]['sha256'],'drafter changed after fixed-K gate')
        a.drafter_sha256=draft[0]['sha256']
    for r,host in enumerate(hosts):
        got=ssh(host,['cat','/proc/sys/kernel/random/boot_id'],timeout=20).stdout.decode().strip()
        require(got==previous['rank_preflight'][r]['boot_id'],'rank rebooted; repeat transport preflight')
    launched=[]
    try:
        for r in list(range(7,0,-1))+[0]:
            host=hosts[r];values=compose.render.rank_values(site,str(r),8)
            name=values['SWITCHLESS_CONTAINER_NAME']
            exists=ssh(host,['docker','inspect',name],check=False,timeout=20)
            require(exists.returncode!=0, 'requested container name is already occupied')
            argv=compose.baked_argv(profile,site,r,image=previous['identity']['image_id'])
            ssh(host,['mkdir','-p',values['SWITCHLESS_JIT_CACHE_DIR'],values['SWITCHLESS_PROFILES_DIR'],
                     values['SWITCHLESS_HF_CACHE_DIR']],timeout=20)
            result=ssh(host,argv,timeout=120)
            container_id=result.stdout.decode().strip()
            require(len(container_id)==64 and all(c in '0123456789abcdef' for c in container_id), 'missing owned container id')
            launched.append((host,container_id))
        print('Started the admitted '+a.stage+' candidate. Run the smoke and memory checks before advancing.')
        return launched
    except BaseException:
        failures=cleanup_owned(launched)
        if failures:print('Owned-container cleanup failed on '+str(len(failures))+' rank(s).',file=sys.stderr)
        raise


SAMPLE_CODE = '''import json,pathlib,sys,threading,time
phase=['before-load'];stop=threading.Event()
def control():
 for line in sys.stdin:
  value=line.strip()
  if value in ('before-load','load','ready','serve'):phase[0]=value
 stop.set()
threading.Thread(target=control,daemon=True).start()
deadline=time.monotonic()+7200
while not stop.is_set() and time.monotonic()<deadline:
 value=int(next(s.split()[1] for s in pathlib.Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
 print(json.dumps(dict(monotonic=time.monotonic(),phase=phase[0],mem_available_bytes=value)),flush=True)
 stop.wait(.2)
'''


def run_stage(a):
    site,topo,profile,hosts,previous=validate_launch(a)
    if not a.go:return launch(a)
    require(a.out is not None,'a new --out directory is required')
    out=a.out.resolve();out.mkdir(parents=True,mode=0o700,exist_ok=False)
    samplers=[];files=[];launched=[]
    record=dict(schema='tp8-test/1',stage=a.stage,ranks=8,ok=False,failures=[],identity=previous['identity'],
                rank_preflight=previous['rank_preflight'],source_receipts_verified=False,metrics=[],routes=previous['routes'],
                source_receipts=[],completion_gate=False,memory_gate=False,speculation_gate=False)
    def phase(value):
        for p in samplers:p.stdin.write((value+'\n').encode());p.stdin.flush()
    try:
        for rank,host in enumerate(hosts):
            dest=(out/f'memory-rank{rank}.jsonl').open('wb');files.append(dest)
            proc=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10',host,
                shlex.join(['python3','-u','-c',SAMPLE_CODE])],stdin=subprocess.PIPE,stdout=dest,stderr=subprocess.DEVNULL)
            samplers.append(proc)
        deadline=time.monotonic()+20
        while not all((out/f'memory-rank{r}.jsonl').stat().st_size for r in range(8)):
            require(time.monotonic()<deadline and all(p.poll() is None for p in samplers),'memory sampler startup failed')
            time.sleep(.1)
        phase('load');time.sleep(.25)
        launched=launch(a)
        import urllib.request
        endpoint='http://'+site.get('SWITCHLESS_API_HOST',site['SWITCHLESS_MASTER_ADDR'])+':'+site['SWITCHLESS_API_PORT']
        deadline=time.monotonic()+a.timeout
        while True:
            require(time.monotonic()<deadline,'model startup timed out')
            require(all(p.poll() is None for p in samplers),'memory recorder exited')
            for host,container in launched:
                state=ssh(host,['docker','inspect','--format','{{.State.Running}}',container],timeout=20).stdout.decode().strip()
                require(state=='true','a model worker exited before ready')
            try:
                with urllib.request.urlopen(endpoint+'/health',timeout=5) as response:
                    if response.status==200:break
            except OSError:pass
            time.sleep(2)
        phase('ready');time.sleep(.5)
        for rank,(host,container) in enumerate(sorted(launched,key=lambda row:hosts.index(row[0]))):
            log=ssh(host,['docker','logs',container],timeout=30)
            text=(log.stdout+log.stderr).decode();(out/f'rank{rank}-startup.log').write_text(text)
            require('GLM53_TP8_CONTRACT rank='+str(rank) in text,'rank startup contract marker missing')
        phase('serve');time.sleep(.25)
        import bench
        result={}
        try:bench.run(endpoint,a.stage,uuid.uuid4().hex,result=result)
        finally:write(out/'bench.json',result)
        require(all(p.poll() is None for p in samplers),'memory sampling stopped before the benchmark ended')
        record.update({k:result[k] for k in ('completion_gate','speculation_gate','metrics')})
        if 'speculation_counts' in result:record['speculation_counts']=result['speculation_counts']
        if hasattr(a,'drafter_sha256'):record['drafter_sha256']=a.drafter_sha256
        record['ok']=result['ok']
    except BaseException as exc:
        record['failures'].append(dict(phase='stage',error_type=type(exc).__name__,error=str(exc)));record['ok']=False
        raise
    finally:
        for proc in samplers:
            try:proc.stdin.close();proc.wait(timeout=5)
            except BaseException:proc.terminate()
        for f in files:f.close()
        failures=cleanup_owned(launched)
        if failures:record['failures'].extend(failures);record['ok']=False
        try:
            samples=[dict(json.loads(line),rank=r) for r in range(8)
                     for line in (out/f'memory-rank{r}.jsonl').read_text().splitlines()]
            record['memory_gate']=memory_gate(samples,a.stage)
            record['memory_min_bytes']={str(r):min(s['mem_available_bytes'] for s in samples if s['rank']==r) for r in range(8)}
        except BaseException as exc:
            record['failures'].append(dict(phase='memory',error=str(exc)));record['ok']=False
        record['source_receipts']=[dict(file=f.name,sha256=sha(f)) for f in sorted(out.iterdir()) if f.is_file()]
        record['source_receipts_verified']=record['ok'];record['finished_unix']=time.time()
        record['qualification']='bring-up smoke only; state byte correctness and performance promotion are separate'
        write(out/'receipt.json',record)
    require(record['ok'],'stage failed; retain the output directory')
    print(out/'receipt.json')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('transport','launch','run-stage','validate-launch'):
        x=sub.add_parser(name);x.add_argument('--site',type=Path,required=True);x.add_argument('--topology',type=Path,required=True)
        x.add_argument('--go',action='store_true');x.add_argument('--out',type=Path)
        if name!='transport':
            x.add_argument('--receipt',type=Path,required=True);x.add_argument('--stage',choices=('target','fixed-k','graphs'),default='target')
        if name=='run-stage':x.add_argument('--timeout',type=int,default=5400)
    x=sub.add_parser('bundle');x.add_argument('--receipt',type=Path,required=True);x.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.command=='transport':
        if a.go and not a.out:p.error('--go requires a new --out directory')
        transport(a)
    elif a.command=='launch':launch(a)
    elif a.command=='run-stage':run_stage(a)
    elif a.command=='validate-launch':validate_launch(a);print('receipt valid')
    else:write(a.out,bundle(verified_receipt(a.receipt)));print(a.out)


if __name__=='__main__':main()
