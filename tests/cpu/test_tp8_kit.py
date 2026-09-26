# SPDX-License-Identifier: Apache-2.0
"""Fault-injected kit receipts, routes, privacy, variants and streaming probes. No node access."""
import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/tp8-test'));sys.path.insert(0,str(ROOT/'fabric'))
import contracts as c
import bench
import ring
import kit


class Receipts(unittest.TestCase):
    def rows(self):
        return [dict(rank=r,ok=True,mem_available_bytes=32*1024**3,**{k:'a'*64 for k in c.IDENTITY_FIELDS}) for r in range(8)]

    def test_rank_identity_is_all_or_nothing(self):
        rows=self.rows();c.identity(rows,8)
        for mutate in (lambda x:x.pop(),lambda x:x[1].update(rank=0),lambda x:x[1].update(nccl_sha256='b'*64),
                       lambda x:x[1].update(mem_available_bytes=15*1024**3),lambda x:x[1].update(ok=False)):
            bad=copy.deepcopy(rows);mutate(bad)
            with self.assertRaises(ValueError):c.identity(bad,8)

    def receipt(self):
        return dict(schema='tp8-test/1',ok=True,stage='transport',ranks=8,failures=[],
                    identity=c.identity(self.rows(),8),finished_unix=100,source_receipts_verified=True)

    def test_no_skipped_failed_stale_or_other_candidate_admission(self):
        receipt=self.receipt();expected=receipt['identity'];c.admit(receipt,'target',expected,now=101)
        for change in ({'ok':False},{'ranks':4},{'finished_unix':-100000},{'stage':'target'},
                       {'failures':['failed']},{'source_receipts_verified':False}):
            with self.assertRaises(ValueError):c.admit(dict(receipt,**change),'target',expected,now=101)
        with self.assertRaises(ValueError):c.admit(receipt,'graphs',expected,now=101)
        with self.assertRaises(ValueError):c.admit(receipt,'target',dict(expected,nccl_sha256='b'*64),now=101)

    def test_memory_requires_continuous_lifecycle_on_every_rank(self):
        samples=[dict(rank=r,phase=p,monotonic=i*.2,mem_available_bytes=32*1024**3)
                 for r in range(8) for i,p in enumerate(('before-load','load','load','ready','serve','serve'))]
        c.memory_gate(samples,'graphs')
        for mutate in (lambda s:s.pop(),lambda s:s[4].update(mem_available_bytes=1),
                       lambda s:s[2].update(monotonic=3),lambda s:s[3].update(phase='load')):
            bad=copy.deepcopy(samples);mutate(bad)
            # Removing only one of two serve samples retains coverage, so remove a complete rank separately.
            if len(bad)<len(samples):bad=[s for s in bad if s['rank']!=7]
            with self.assertRaises(ValueError):c.memory_gate(bad,'graphs')

    def test_bundle_preserves_hashes_and_endpoint_relationships_but_drops_secrets(self):
        secret='hf'+'_'+'x'*40
        rec=self.receipt();rec['identity'][secret+'_sha256']='b'*64
        rec.update(environment={'AUTH':secret},headers={'Authorization':secret},logs=secret,
                   metrics=[{'tokens':42,'output':secret,'decode_tps':12.5,'bad':float('nan')}],
                   routes=[dict(rank=0,peer_rank=1,local='192.0.2.1',remote='192.0.2.2'),
                           dict(rank=1,peer_rank=0,local='192.0.2.2',remote='192.0.2.1')])
        clean=c.bundle(rec);text=json.dumps(clean)
        self.assertNotIn(secret,text);self.assertNotIn('192.0.2.',text)
        self.assertEqual(clean['identity']['nccl_sha256'],'a'*64)
        self.assertEqual(clean['metrics'],[dict(tokens=42,decode_tps=12.5)])
        self.assertEqual(clean['routes'][0]['local'],clean['routes'][1]['remote'])
        with self.assertRaises(ValueError):c.bundle(dict(rec,ranks=secret))

    def test_receipt_requires_unchanged_raw_files_inside_its_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw.json';raw.write_text('{"ok": true}')
            receipt=root/'receipt.json'
            kit.write(receipt,dict(source_receipts=[dict(file='raw.json',sha256=kit.sha(raw))]))
            kit.verified_receipt(receipt)
            raw.write_text('{"ok": false}')
            with self.assertRaises(ValueError):kit.verified_receipt(receipt)
            kit.write(receipt,dict(source_receipts=[dict(file='../outside.json',sha256='a'*64)]))
            with self.assertRaises(ValueError):kit.verified_receipt(receipt)

    def test_cleanup_continues_after_a_stop_timeout_and_uses_only_owned_ids(self):
        ids=['1'*64,'2'*64];calls=[]
        def ssh(host,argv,**kw):
            calls.append((host,argv))
            if argv[1]=='stop':raise subprocess.TimeoutExpired(argv,30)
            return subprocess.CompletedProcess(argv,0)
        with patch.object(kit,'ssh',side_effect=ssh):
            self.assertEqual(kit.cleanup_owned(list(zip(('rank0','rank1'),ids))),[])
        self.assertEqual([args[-1] for _,args in calls if args[1]=='rm'],list(reversed(ids)))
        self.assertTrue(all(args[-1] in ids for _,args in calls))


class Routes(unittest.TestCase):
    def test_preflight_checks_exact_endpoints_and_distinct_pci_roots(self):
        preflight=kit.module('tp8_preflight','fabric/preflight.py');topo=ring.generate(8)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);addresses={h:a for edge in topo['links'] if 0 in edge['ranks']
                                    for h,a in edge['addresses']['0'].items()}
            for hca,address in addresses.items():
                h=root/hca;port=h/'ports/1';port.mkdir(parents=True)
                domain='0002' if hca.startswith('roceP2') else '0000'
                pci=root/(domain+':01:00.'+hca[-1]);pci.mkdir(exist_ok=True);(h/'device').symlink_to(pci)
                for name,value in {'state':'4: ACTIVE','link_layer':'Ethernet','gids/3':'::ffff:'+address,
                                   'gid_attrs/types/3':'RoCE v2','gid_attrs/ndevs/3':'eth0'}.items():
                    path=port/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
            self.assertTrue(preflight.check(root,topo,0,3)['ok'])
            device=root/'roceP2p1s0f0/device';original=device.resolve();device.unlink()
            device.symlink_to(root/'0000:01:00.0')
            self.assertFalse(preflight.check(root,topo,0,3)['ok'])
            device.unlink();device.symlink_to(original)
            (root/'rocep1s0f0/ports/1/gids/3').write_text('::ffff:192.0.2.17')
            self.assertFalse(preflight.check(root,topo,0,3)['ok'])

    def log(self):
        topo=ring.generate(8);rank=0;rows=['Ring 00 : 7 -> 0 -> 1',
           'Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0',
           'Channel 00/0 : 7[0] -> 0[0] [receive] via NET/IB/0',
           'AllReduce: 128 Bytes -> Algo Ring proto Simple channel{Lo..Hi}={0..0}',
           'AllGather: 128 Bytes -> Algo Ring proto Simple channel{Lo..Hi}={0..0}']
        import ipaddress
        for edge in topo['links']:
            if 0 not in edge['ranks']:continue
            peer=next(r for r in edge['ranks'] if r!=0)
            for hca,local in edge['addresses']['0'].items():
                remote=edge['addresses'][str(peer)][hca]
                gid=lambda x:f'{int(ipaddress.IPv6Address("::ffff:"+x)):032x}'
                rows.append('NET/IB RouteFinal side='+('send' if peer==1 else 'recv')+' hca='+hca+
                            ' localGid='+gid(local)+' remoteGid='+gid(remote))
        return topo,'\n'.join(rows)

    def test_requires_actual_routes_and_algorithms(self):
        topo,text=self.log();c.route_evidence(text,topo,0)
        for bad in (text.replace('7 -> 0 -> 1','6 -> 0 -> 1'),text.replace('NET/IB/0','NET/Socket/0'),
                    text.replace('Algo Ring','Algo Tree'),text.replace('NET/IB RouteFinal','unobserved'),
                    text.replace('roceP2p1s0','rocep1s0')):
            with self.assertRaises(ValueError):c.route_evidence(bad,topo,0)
        with self.assertRaises(ValueError):c.route_evidence('NCCL_ALGO=Ring\nNCCL_NET=IB',topo,0)


class Variants(unittest.TestCase):
    def test_both_api_names_survive_every_stage_rank_and_launch_format(self):
        site=json.loads((ROOT/'launch/profiles/tp8/canonical-launch.json').read_text())['site']
        site['SWITCHLESS_DRAFTER_REPO_DIR']='/srv/drafter'
        profiles={'target':kit.compose.load_profile('tp8')}
        for stage in ('fixed-k','graphs'):
            profiles[stage]=json.loads((ROOT/'launch/variants/tp8'/ (stage+'.json')).read_text())['profile']
        # Pinned vLLM ModelConfig accepts str | list[str] | None; its CLI uses nargs='+'.
        parser=argparse.ArgumentParser(add_help=False)
        parser.add_argument('--served-model-name',nargs='+')
        parser.add_argument('--max-model-len',type=int)
        for stage,profile in profiles.items():
            for rank in range(8):
                overlay=kit.compose.render.render(profile,site,rank)
                baked=kit.compose.baked_argv(profile,site,rank)
                project,_=kit.compose.project(profile,site,rank,kit.compose.CONTEXT)
                command=project['services'][kit.compose.SERVICE]['command']
                for form,argv in (('overlay',overlay),('baked',baked),('compose',command)):
                    with self.subTest(stage=stage,rank=rank,form=form):
                        args,_=parser.parse_known_args(argv)
                        self.assertEqual(args.served_model_name,
                            ['RedHatAI/GLM-5.3-Flash-NVFP4','nvidia/GLM-5.3-Flash-NVFP4'])
                        self.assertEqual(args.max_model_len,8192)
                        self.assertEqual(argv.count('--served-model-name'),1)

    def test_launch_identity_binds_mounts_flags_and_destinations(self):
        site=json.loads((ROOT/'launch/profiles/tp8/canonical-launch.json').read_text())['site']
        profile=kit.compose.load_profile('tp8');hosts=site['SWITCHLESS_RANK_HOST_IPS'].split(',')
        baseline=kit.launch_config(site,profile,hosts)
        for key,value in [('SWITCHLESS_MODEL_DIR','/srv/another-model'),
                          ('SWITCHLESS_TOKENIZER_FILE','/srv/another-tokenizer.json'),
                          ('SWITCHLESS_DRAFTER_REPO_DIR','/srv/another-drafter'),
                          ('SWITCHLESS_BUILD_IMAGE','another:tag')]:
            self.assertNotEqual(baseline,kit.launch_config(dict(site,**{key:value}),profile,hosts))
        self.assertNotEqual(baseline,kit.launch_config(site,profile,list(reversed(hosts))))

    def test_variants_reproduce_through_the_importer_and_leave_base_unchanged(self):
        importer=kit.module('variant_import','sources/import_plan.py')
        before=(ROOT/'launch/profiles/tp8/profile.json').read_bytes()
        variants=importer.tp8_variants(ROOT)
        site=json.loads((ROOT/'launch/profiles/tp8/canonical-launch.json').read_text())['site']
        site['SWITCHLESS_DRAFTER_REPO_DIR']='/srv/drafter'
        for stage,variant in variants.items():
            self.assertEqual(variant,json.loads((ROOT/'launch/variants/tp8'/ (stage+'.json')).read_text()))
            p=variant['profile']
            for rank in range(8):
                argv=kit.compose.baked_argv(p,site,rank)
                spec=json.loads(argv[argv.index('--speculative-config')+1]);self.assertEqual(spec['num_speculative_tokens'],4)
                self.assertIn('GLM53_SPEEDUP_MODE=off',argv)
            self.assertEqual('--enforce-eager' in p['argv_template'],stage=='fixed-k')
        self.assertEqual((ROOT/'launch/profiles/tp8/profile.json').read_bytes(),before)


class Streaming(unittest.TestCase):
    def response(self,finish='stop',usage=True):
        events=[{'choices':[{'delta':{'content':'42'},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':finish}]}]
        if usage:events.append({'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':2}})
        return io.BytesIO(('\n'.join('data: '+json.dumps(e)+'\n' for e in events)+'data: [DONE]\n').encode())

    def test_receipt_needs_stop_and_actual_usage(self):
        with patch('urllib.request.urlopen',return_value=self.response()):
            row=bench.completion('http://example.invalid','probe');self.assertEqual(row['text'],'42')
            self.assertEqual(row['prompt_tokens'],10);self.assertGreater(row['elapsed_ms'],0)
        for response in (self.response(finish='length'),self.response(usage=False)):
            with patch('urllib.request.urlopen',return_value=response), self.assertRaises(ValueError):
                bench.completion('http://example.invalid','probe')


if __name__=='__main__':unittest.main()
