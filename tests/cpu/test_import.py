# SPDX-License-Identifier: Apache-2.0
"""sources/import_plan.py on a copy of the repository with a synthetic plan: private site values never reach the
repository (in any output, including src/ and the chat template), and a failed import leaves the tree as it was."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DP = '/usr/local/lib/python3.12/dist-packages/'
IMAGE = 'sha256:017fd0ba0a265dd81b78b352b14d33df03e9996f8f45870d4c3df84e08b1ebcb'
PRIVATE_ADDR = '198.51.100.7'          # stands in for a lab address (documentation range, so this file stays clean)
PRIVATE_DIR = '/srv/lab-private/models'  # stands in for a lab host path
PRE = b'# SPDX-License-Identifier: Apache-2.0\ndef f():\n    return 1\n'
POST = b'# SPDX-License-Identifier: Apache-2.0\ndef f():\n    return 2\n'
OURS = b'VALUE = 3\n'
TEMPLATE = b'{{ messages }}\n'
OVERRIDE = b'{"mode": "off"}\n'
COPY = ('sources.lock.json', 'sources', 'patches', 'src', 'launch', 'release', 'licenses', 'LICENSE', 'NOTICE', 'build')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree_digest(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:
            continue
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b'\0')
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


class Import(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.root, self.work = t / 'recipe', t / 'work'
        self.root.mkdir()
        self.work.mkdir()
        for name in COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copyfile(src, self.root / name)

    def document(self, ours=OURS, template=TEMPLATE, post=POST):
        w = self.work
        for name, data in (('ours.py', ours), ('pre.py', PRE), ('post.py', post), ('tmpl.jinja', template),
                           ('override.json', OVERRIDE)):
            (w / name).write_bytes(data)
        argv = {}
        for r in ('0', '1'):
            argv[r] = ['docker', 'run', '--name', f'labbox-{r}', '-e', f'VLLM_HOST_IP=198.51.100.{10 + int(r)}',
                       '--mount', f'type=bind,src={PRIVATE_DIR},dst=/models/target,readonly',
                       '--mount', f'type=bind,src=/stage/ours.py,dst={DP}ours.py,readonly',
                       '--mount', f'type=bind,src=/stage/mod.py,dst={DP}vllm/mod.py,readonly',
                       '--mount', 'type=bind,src=/stage/tmpl.jinja,dst=/chat_template_mm.jinja,readonly',
                       '--mount', 'type=bind,src=/stage/runtime,dst=/opt/glm53-speedup/runtime,readonly',
                       IMAGE, '/models/target', '--node-rank', r, '--master-addr', PRIVATE_ADDR, '--master-port', '25104']
            argv[r] += ['--host', '0.0.0.0', '--port', '8888'] if r == '0' else ['--headless']
        return {
            'schema': 'switchless-import/1', 'topology': 'tp2',
            'plan': {'sha256': 'b' * 64, 'label': 'synthetic', 'deterministic': None, 'nondeterministic_note': None},
            'image': {'id': IMAGE, 'tag': 'local/test'},
            'argv': argv,
            'site': {'values': {'SWITCHLESS_MASTER_ADDR': PRIVATE_ADDR, 'SWITCHLESS_MASTER_PORT': '25104',
                                'SWITCHLESS_API_PORT': '8888', 'SWITCHLESS_MODEL_DIR': PRIVATE_DIR},
                     'per_rank': {r: {'SWITCHLESS_RANK_HOST_IP': f'198.51.100.{10 + int(r)}',
                                      'SWITCHLESS_CONTAINER_NAME': f'labbox-{r}'} for r in ('0', '1')},
                     'mount_sources': {'/models/target': 'SWITCHLESS_MODEL_DIR'},
                     'env': {'VLLM_HOST_IP': 'SWITCHLESS_RANK_HOST_IP'},
                     'args': {'--master-addr': 'SWITCHLESS_MASTER_ADDR', '--master-port': 'SWITCHLESS_MASTER_PORT',
                              '--port': 'SWITCHLESS_API_PORT'}},
            'files': [
                {'target': DP + 'ours.py', 'sha256': sha(ours), 'local': str(w / 'ours.py'), 'kind': 'src',
                 'component': 'ours', 'licence': 'Apache-2.0', 'attribution': [], 'evidence': 'test'},
                {'target': DP + 'vllm/mod.py', 'sha256': sha(post), 'local': str(w / 'post.py'), 'kind': 'patch',
                 'component': 'vllm', 'licence': 'Apache-2.0', 'attribution': [], 'evidence': 'test',
                 'preimage': {'sha256': sha(PRE), 'local': str(w / 'pre.py'), 'evidence': 'test', 'origin': 'test',
                              'upstream_relation': 'test file'}},
                {'target': '/chat_template_mm.jinja', 'sha256': sha(template), 'local': str(w / 'tmpl.jinja'),
                 'kind': 'template', 'licence': 'MIT', 'attribution': [], 'evidence': 'test',
                 'preimage': {'sha256': sha(b'x'), 'local': str(w / 'tmpl.jinja'), 'evidence': 'test', 'origin': 'test'}},
            ],
            'dirs': [{'target': '/opt/glm53-speedup/runtime', 'kind': 'config-dir', 'licence': 'Apache-2.0',
                      'attribution': [], 'files': {'override.json': {'sha256': sha(OVERRIDE),
                                                                     'local': str(w / 'override.json')}}}],
            'binaries': [],
        }

    def run_import(self, doc):
        path = self.work / 'import.json'
        path.write_text(json.dumps(doc))
        return subprocess.run([sys.executable, '-B', str(ROOT / 'sources/import_plan.py'), str(path),
                               '--root', str(self.root)], capture_output=True, text=True)

    def assert_refused_and_unchanged(self, doc, message):
        before = tree_digest(self.root)
        r = self.run_import(doc)
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn(message, r.stderr)
        self.assertEqual(tree_digest(self.root), before, 'a failed import changed the repository')
        self.assertEqual([p.name for p in self.root.iterdir() if p.name.startswith('.import-')], [])

    def test_clean_import_writes_the_profile(self):
        r = self.run_import(self.document())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual((self.root / 'src/tp2/ours.py').read_bytes(), OURS)
        self.assertTrue((self.root / 'patches/vllm/tp2/mod.py.patch').is_file())
        self.assertFalse((self.root / 'patches/b12x/tp2').exists())   # the old profile's other patches are gone
        profile = json.loads((self.root / 'launch/profiles/tp2/profile.json').read_text())
        fixture = json.loads((self.root / 'launch/profiles/tp2/canonical-launch.json').read_text())
        self.assertEqual(profile['source_plan']['launch_sha256'], fixture['launch_sha256'])
        for p in self.root.rglob('*'):
            if p.is_file():
                data = p.read_bytes()
                self.assertNotIn(PRIVATE_ADDR.encode(), data, p)
                self.assertNotIn(PRIVATE_DIR.encode(), data, p)

    def test_derived_plan_keeps_its_derivation_and_qualification(self):
        doc = self.document()
        doc['plan']['derived_from'] = {'plan_sha256': 'a' * 64, 'note': 'the base plan with one change', 'steps': []}
        doc['plan']['qualification'] = {'plan_sha256': 'b' * 64, 'evidence': [{'change': 'all', 'result': 'checked'}],
                                        'not_established': ['not benchmarked']}
        r = self.run_import(doc)
        self.assertEqual(r.returncode, 0, r.stderr)
        profile = json.loads((self.root / 'launch/profiles/tp2/profile.json').read_text())
        self.assertEqual(profile['status'], 'derived-from-measured-lab-plan')
        self.assertEqual(profile['source_plan']['derived_from'], doc['plan']['derived_from'])
        self.assertEqual(profile['source_plan']['qualification'], doc['plan']['qualification'])
        patch = (self.root / 'patches/vllm/tp2/mod.py.patch').read_text()
        self.assertTrue(patch.startswith('RiNGSiDE overlay patch, profile tp2\n'), patch[:60])

    def test_leak_in_a_src_file_is_refused(self):
        self.assert_refused_and_unchanged(self.document(ours=b'MASTER = "%s"\n' % PRIVATE_ADDR.encode()),
                                          'private site value would reach the repository: src/tp2/ours.py')

    def test_leak_in_the_chat_template_is_refused(self):
        self.assert_refused_and_unchanged(self.document(template=b'{# %s #}\n' % PRIVATE_DIR.encode()),
                                          'launch/profiles/tp2/chat_template_mm.jinja')

    def test_container_name_leak_is_refused(self):
        self.assert_refused_and_unchanged(self.document(ours=b'NAME = "labbox-1"\n'), 'private site value')

    def test_private_path_pattern_is_refused(self):
        home = b'/' + b'home/someone/x'   # split so that this file does not match the hygiene pattern itself
        self.assert_refused_and_unchanged(self.document(ours=b'ROOT = "' + home + b'"\n'), 'private path or address')

    def test_failure_after_the_first_output_leaves_the_old_profile(self):
        doc = self.document()
        doc['files'][1]['sha256'] = sha(b'not the served file')   # the local copy is not the served file
        self.assert_refused_and_unchanged(doc, 'local copy differs from the served sha256')

    def test_patch_that_does_not_reproduce_the_served_file_is_refused(self):
        doc = self.document()
        (self.work / 'pre-other.py').write_bytes(PRE + b'# another image\n')
        doc['files'][1]['preimage'].update(local=str(self.work / 'pre-other.py'), sha256=sha(PRE))
        self.assert_refused_and_unchanged(doc, 'preimage copy differs')

    def test_render_mismatch_leaves_the_old_profile(self):
        doc = self.document()
        doc['argv']['1'].insert(2, '--privileged')   # rank 1 differs from rank 0 beyond the site values
        self.assert_refused_and_unchanged(doc, 'differs from rank 0')


if __name__ == '__main__':
    unittest.main()
