# SPDX-License-Identifier: Apache-2.0
"""The TP4 release plan's served changes against the repository's other records: the rowspread top-k extension's
build inputs and artefacts, the credits of the upstream work its steps port, and the variables its steps set or
leave unset."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
DP = '/usr/local/lib/python3.12/dist-packages/'
ROWSPREAD = ROOT / 'build/topk_rowspread'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def profile():
    return json.loads((ROOT / 'launch/profiles/tp4/profile.json').read_text())


def served(topo='tp4'):
    return {i['target']: i for i in json.loads((ROOT / 'sources/installed-files.json').read_text())['profiles'][topo]['files']}


def assets():
    return {a['name']: a for a in json.loads((ROOT / 'release/assets.json').read_text())['artefacts']}


class Rowspread(unittest.TestCase):
    def setUp(self):
        if DP + 'nvfp4_topk_rowspread' not in served():
            self.skipTest('the TP4 profile serves no rowspread extension')

    def test_build_inputs_are_the_pinned_sources(self):
        build = load('rowspread_build', 'build/topk_rowspread/build.py')
        for name, want in build.OWN.items():
            self.assertEqual(sha(ROWSPREAD / name), want, name)
        for rel, want in build.STAGE15.items():
            self.assertEqual(sha(ROOT / 'docker/base/15-topk-repair/src' / rel), want, rel)

    def test_recipe_sources_differ_from_the_measured_build_only_in_the_notice(self):
        receipt = json.loads((ROWSPREAD / 'measured-build-receipt.json').read_text())
        lines = (ROWSPREAD / 'persistent_topk.cuh').read_bytes().split(b'\n')
        self.assertTrue(lines[0].startswith(b'// Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): '), lines[0])
        self.assertTrue(lines[1].startswith(b'// '))
        self.assertEqual(hashlib.sha256(b'\n'.join(lines[2:])).hexdigest(), receipt['persistent_topk_sha256'])
        self.assertEqual(sha(ROWSPREAD / 'bindings.cu'), receipt['binding_sha256'])
        self.assertEqual(sha(ROOT / 'docker/base/15-topk-repair/src/csrc/libtorch_stable/topk.cu'), receipt['topk_cu_sha256'])
        self.assertEqual(sha(ROOT / 'src/tp4/nvfp4_topk_rowspread/__init__.py'), receipt['loader_sha256'])

    def test_artefacts_are_the_measured_build(self):
        item = served()[DP + 'nvfp4_topk_rowspread']
        receipt_bytes = (ROWSPREAD / 'measured-build-receipt.json').read_bytes()
        receipt = json.loads(receipt_bytes)
        a = assets()
        self.assertEqual(item['files']['_kernel.so']['sha256'], receipt['binary_sha256'])
        self.assertEqual(a['nvfp4-topk-rowspread']['sha256'], receipt['binary_sha256'])
        self.assertEqual(item['files']['build-receipt.json']['sha256'], hashlib.sha256(receipt_bytes).hexdigest())
        self.assertEqual(a['nvfp4-topk-rowspread-receipt']['sha256'], hashlib.sha256(receipt_bytes).hexdigest())
        self.assertEqual(item['files']['__init__.py'].get('source'), 'src/tp4/nvfp4_topk_rowspread/__init__.py')
        self.assertEqual({f.get('artefact') for f in item['files'].values()},
                         {None, 'nvfp4-topk-rowspread', 'nvfp4-topk-rowspread-receipt'})

    def test_the_loader_checks_what_the_build_writes(self):
        loader = (ROOT / 'src/tp4/nvfp4_topk_rowspread/__init__.py').read_text()
        for key in re.findall(r'_RECEIPT\["(\w+)"\]', loader):
            self.assertIn(f'{key}=', (ROWSPREAD / 'build.py').read_text().replace("'", '').replace(' ', ''), key)

    def test_the_image_build_makes_both_artefacts(self):
        text = (ROOT / 'docker/Dockerfile').read_text()
        self.assertIn('FROM base AS topk-rowspread', text)
        self.assertIn('build/topk_rowspread/build.py --src /opt/topk-repair/src', text.replace('/src/topk_rowspread/build.py', 'build/topk_rowspread/build.py'))
        self.assertIn('COPY --from=topk-rowspread /artefacts/ /artefacts/', text)


class Steps(unittest.TestCase):
    def steps(self):
        derived = profile()['source_plan'].get('derived_from') or {}
        return derived.get('steps') or []

    def test_ported_upstream_work_is_credited(self):
        notice = (ROOT / 'NOTICE').read_text()
        credits = (ROOT / 'docs/credits.md').read_text()
        for step in self.steps():
            for credit in step.get('credits', []):
                for ref in re.findall(r'[\w.-]+/[\w.-]+#\d+|pull request \d+', credit):
                    number = ref.split('#')[-1].split()[-1]
                    self.assertIn(number, notice, (step['step'], ref))
                    self.assertIn(number, credits, (step['step'], ref))
                for owner in re.findall(r'\b([A-Z][\w-]+/[\w.-]+) [0-9a-f]{7}', credit):
                    self.assertIn(owner, notice, (step['step'], owner))
                    self.assertIn(owner, credits, (step['step'], owner))

    def test_variables_the_plan_leaves_unset_are_documented_as_unset(self):
        argv = profile()['argv_template']
        env = {argv[i + 1].split('=', 1)[0] for i in range(len(argv) - 1) if argv[i] == '-e'}
        doc = (ROOT / 'docs/configuration.md').read_text()
        for step in self.steps():
            for e in step.get('environment', []):
                if e.get('asserted_unset'):
                    self.assertNotIn(e['key'], env)
                    self.assertIn(f"| `{e['key']}` | - | - |", doc, e['key'])
                else:
                    self.assertIn(e['key'], env)

    def test_every_step_names_served_targets(self):
        targets = {t[len(DP):] if t.startswith(DP) else t for t in served()}
        for step in self.steps():
            for t in step.get('targets', []):
                self.assertIn(t, targets, (step['step'], t))


if __name__ == '__main__':
    unittest.main()
