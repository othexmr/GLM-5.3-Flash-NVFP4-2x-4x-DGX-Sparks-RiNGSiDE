# SPDX-License-Identifier: Apache-2.0
"""TP8 construction, startup rejection and untouched release-byte contracts. CPU only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fabric'))
from ring import generate, validate


def module(name, path):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    return m


guard = module('tp8_guard', 'src/tp8/glm53_tp8.py')


def config():
    return NS(parallel_config=NS(tensor_parallel_size=8, pipeline_parallel_size=1, data_parallel_size=1,
        decode_context_parallel_size=1, prefill_context_parallel_size=1, enable_expert_parallel=False,
        enable_eplb=False, use_sequence_parallel_moe=False),
        model_config=NS(enforce_eager=True, max_model_len=8192, hf_config=NS(hidden_size=4096,
            num_attention_heads=64, linear_num_heads=64, index_n_heads=32, kv_lora_rank=512,
            q_lora_rank=1536, vocab_size=154880)),
        scheduler_config=NS(max_num_seqs=1), cache_config=NS(kv_cache_memory_bytes=4*1024**3),
        num_speculative_tokens=0)


class Contract(unittest.TestCase):
    def env(self):
        return dict(guard.FIXED_ENV, NCCL_SWITCHLESS_CABLE_SUBNETS=','.join(x['subnet'] for x in generate(8)['links']))

    def test_actual_config_dimensions_are_owned_correctly(self):
        record = guard.inspect(config(), self.env())
        self.assertEqual((record['local_mla_heads'], record['local_kda_heads'], record['local_indexer_heads'],
                          record['residual_hidden']), (8, 8, 32, 4096))
        self.assertEqual(record['status'], 'GPU_UNQUALIFIED')

    def test_every_unqualified_lever_is_rejected(self):
        for flag in guard.ZERO_SWITCHES:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                guard.inspect(config(), dict(self.env(), **{flag: '1'}))

    def test_required_transport_settings_cannot_be_missing_or_changed(self):
        for flag in guard.FIXED_ENV:
            for changed in (False, True):
                env = self.env()
                if changed: env[flag] = 'unqualified'
                else: del env[flag]
                with self.subTest(flag=flag, changed=changed), self.assertRaises(ValueError):
                    guard.inspect(config(), env)

    def test_bad_geometry_stage_and_memory_rejected(self):
        for mutate in (lambda c: setattr(c.parallel_config, 'tensor_parallel_size', 4),
                       lambda c: setattr(c.model_config.hf_config, 'index_n_heads', 4),
                       lambda c: setattr(c.cache_config, 'kv_cache_memory_bytes', 32*1024**3),
                       lambda c: setattr(c.model_config, 'max_model_len', 32768),
                       lambda c: setattr(c.scheduler_config, 'max_num_seqs', 2),
                       lambda c: setattr(c, 'num_speculative_tokens', 4)):
            c = config(); mutate(c)
            with self.assertRaises(ValueError): guard.inspect(c, self.env())
        with self.assertRaises(ValueError):
            guard.inspect(config(), dict(self.env(), NCCL_SWITCHLESS_CABLE_SUBNETS=','.join(['192.0.2.0/24']*8)))

    def test_explicit_fixed_k_and_graph_stages(self):
        c = config(); c.num_speculative_tokens = 4
        guard.inspect(c, dict(self.env(), GLM53_TP8_STAGE='fixed-k'))
        c.model_config.enforce_eager = False; c.scheduler_config.max_num_seqs = 4
        guard.inspect(c, dict(self.env(), GLM53_TP8_STAGE='graphs'))


class Ring(unittest.TestCase):
    def test_all_four_and_eight_routes(self):
        for n in (4, 8):
            t = generate(n); adj = validate(t)
            for r in range(n): self.assertEqual(adj[r], {(r-1)%n, (r+1)%n})

    def test_bad_topology_rejected(self):
        for change in (lambda t: t['nodes'][0].update(rank=True),
                       lambda t: t['nodes'][1].update(management_address=t['nodes'][0]['management_address']),
                       lambda t: t['links'][1].update(subnet=t['links'][0]['subnet']),
                       lambda t: t['links'][1].update(port='f1'),
                       lambda t: t['links'][0]['addresses'].update({'7': {}}),
                       lambda t: t['links'][0].update(ranks=[0,0])):
            t = generate(8); change(t)
            with self.assertRaises(ValueError): validate(t)
        for n in (2, 6, 7, 9):
            with self.assertRaises(ValueError): generate(n)

    def test_disconnected_even_cycles_rejected(self):
        t = generate(8)
        for link, pair in zip(t['links'], ([0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4])):
            old = list(link['addresses'].values()); link['ranks'] = pair
            link['addresses'] = dict(zip(map(str, pair), old))
        with self.assertRaisesRegex(ValueError, 'disconnected'): validate(t)


class FrozenRelease(unittest.TestCase):
    def test_release_files_and_manifest_subobjects_unchanged(self):
        fixture = json.loads((ROOT/'tests/fixtures/tp8-frozen-release.json').read_text())
        for path, sha in fixture['files'].items():
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), sha, path)
        for path, entries in fixture['objects'].items():
            doc = json.loads((ROOT/path).read_text())
            for keys, sha in entries:
                sub = doc
                for key in keys: sub = sub[key]
                self.assertEqual(hashlib.sha256(json.dumps(sub,sort_keys=True).encode()).hexdigest(),sha,(path,keys))


if __name__ == '__main__': unittest.main()
