# SPDX-License-Identifier: Apache-2.0
"""launch/render.py renders complete per-rank commands from a site file and never runs them."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('render', ROOT / 'launch/render.py')
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def profile(topo):
    return json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())


class Render(unittest.TestCase):
    def site(self, extra=''):
        tmp = tempfile.NamedTemporaryFile('w', suffix='.env', delete=False)
        self.addCleanup(Path(tmp.name).unlink)
        tmp.write((ROOT / 'site.env.example').read_text() + extra)
        tmp.close()
        return render.read_site(tmp.name), tmp.name

    TP2 = '\nSWITCHLESS_RANK_HOST_IPS=192.0.2.20,192.0.2.21\nSWITCHLESS_MASTER_ADDR=192.0.2.20\n'

    def test_every_rank_renders_without_placeholders(self):
        for topo in ('tp4', 'tp2'):
            site, _ = self.site(self.TP2 if topo == 'tp2' else '')
            p = profile(topo)
            for rank in range(p['ranks']):
                argv = render.render(p, site, rank)
                self.assertEqual(argv[:2], ['docker', 'run'])
                self.assertFalse([t for t in argv if '${' in t], (topo, rank))
                self.assertIn(p['image']['id'], argv)
                self.assertEqual(argv[argv.index('--node-rank') + 1], str(rank))
                self.assertEqual('--headless' in argv, rank != 0)
                self.assertEqual('--port' in argv, rank == 0)
                name = argv[argv.index('--name') + 1]
                self.assertTrue(name.endswith(f'-rank{rank}'))

    def test_overlay_mounts_point_into_the_overlay_root(self):
        site, _ = self.site()
        argv = render.render(profile('tp4'), site, 0)
        mounts = [argv[i + 1] for i, t in enumerate(argv) if t == '--mount']
        overlay = [m for m in mounts if 'src=' + site['SWITCHLESS_OVERLAY_ROOT'] in m]
        # the site mounts: the profile's mount table entries whose source is a site value, not the overlay
        site_mounts = [d for d, m in profile('tp4')['mounts'].items() if not m['source'].startswith('${OVERLAY_ROOT}')]
        self.assertGreaterEqual(len(site_mounts), 5)
        self.assertEqual(len(overlay), len(mounts) - len(site_mounts))
        for m in overlay:
            kv = dict(p.split('=', 1) for p in m.split(',') if '=' in p)
            self.assertEqual(kv['src'], site['SWITCHLESS_OVERLAY_ROOT'] + kv['dst'])
            self.assertTrue(m.endswith(',readonly'))

    def test_per_rank_values(self):
        site, _ = self.site()
        ips = site['SWITCHLESS_RANK_HOST_IPS'].split(',')
        for rank in range(4):
            argv = render.render(profile('tp4'), site, rank)
            self.assertIn('VLLM_HOST_IP=' + ips[rank], argv)
            self.assertTrue(any(t.startswith(f"type=bind,src={site['SWITCHLESS_JIT_CACHE_ROOT']}/rank{rank},dst=/root/.cache")
                                for t in argv))

    def test_rank_address_count_must_match(self):
        site, _ = self.site()
        with self.assertRaisesRegex(ValueError, 'lists 4 addresses'):
            render.render(profile('tp2'), site, 0)

    def test_missing_site_value_is_an_error(self):
        site, _ = self.site(self.TP2)
        del site['SWITCHLESS_MASTER_ADDR']
        with self.assertRaisesRegex(ValueError, 'SWITCHLESS_MASTER_ADDR'):
            render.render(profile('tp2'), site, 0)

    def test_image_override(self):
        site, _ = self.site(self.TP2 + 'SWITCHLESS_IMAGE=registry.example/glm53@sha256:' + 'c' * 64 + '\n')
        argv = render.render(profile('tp2'), site, 1)
        self.assertIn('registry.example/glm53@sha256:' + 'c' * 64, argv)
        self.assertNotIn(profile('tp2')['image']['id'], argv)

    def test_floating_image_reference_is_refused(self):
        for image in ('registry.example/glm53:latest', 'registry.example/glm53', 'sha256:abc',
                      'registry.example/glm53@sha256:' + 'C' * 64):
            site, _ = self.site(self.TP2 + f'SWITCHLESS_IMAGE={image}\n')
            with self.assertRaisesRegex(ValueError, 'pinned by digest'):
                render.render(profile('tp2'), site, 0)
        site, _ = self.site(self.TP2 + 'SWITCHLESS_IMAGE=sha256:' + 'd' * 64 + '\n')
        self.assertIn('sha256:' + 'd' * 64, render.render(profile('tp2'), site, 0))

    def test_matching_quotes_are_stripped(self):
        site, _ = self.site('SWITCHLESS_MODEL_DIR="/srv/models/GLM #5"   # quoted: the # is part of the path\n'
                            "SWITCHLESS_CONTAINER_PREFIX='glm53 tp4'\n")
        self.assertEqual(site['SWITCHLESS_MODEL_DIR'], '/srv/models/GLM #5')
        self.assertEqual(site['SWITCHLESS_CONTAINER_PREFIX'], 'glm53 tp4')
        argv = render.render(profile('tp4'), site, 0)
        self.assertIn('type=bind,src=/srv/models/GLM #5,dst=/models/target,readonly', argv)
        self.assertFalse([t for t in argv if '"' in t and '/srv/' in t])

    def test_unbalanced_quotes_are_an_error(self):
        with self.assertRaisesRegex(ValueError, 'unbalanced quotes'):
            self.site('SWITCHLESS_MODEL_DIR="/srv/models\n')

    def test_comma_in_a_mounted_path_is_refused(self):
        for key in ('SWITCHLESS_MODEL_DIR', 'SWITCHLESS_OVERLAY_ROOT', 'SWITCHLESS_JIT_CACHE_ROOT'):
            site, _ = self.site(f'{key}=/srv/a,readonly=false\n')
            with self.assertRaisesRegex(ValueError, 'must not contain a comma'):
                render.render(profile('tp4'), site, 0)

    def test_canonical_launch_fixture_renders_exactly(self):
        for topo in ('tp4', 'tp2'):
            p = profile(topo)
            fixture = json.loads((ROOT / 'launch/profiles' / topo / 'canonical-launch.json').read_text())
            self.assertEqual(fixture['source_plan_sha256'], p['source_plan']['sha256'])
            self.assertEqual(fixture['launch_sha256'], p['source_plan']['launch_sha256'])
            for rank in range(p['ranks']):
                self.assertEqual(render.render(p, fixture['site'], rank), fixture['argv'][str(rank)], (topo, rank))

    def test_cli_prints_workers_first(self):
        _, path = self.site()
        r = subprocess.run([sys.executable, '-B', str(ROOT / 'launch/render.py'), '--profile', 'tp4', '--site', path],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        ranks = re.findall(r'^# rank (\d)', r.stdout, re.M)
        self.assertEqual(ranks, ['3', '2', '1', '0'])


if __name__ == '__main__':
    unittest.main()
