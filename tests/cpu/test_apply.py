# SPDX-License-Identifier: Apache-2.0
"""sources/apply.py on a synthetic repository: hashes are checked before and after every patch."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('apply', ROOT / 'sources/apply.py')
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)
DP = '/usr/local/lib/python3.12/dist-packages/'
PRE = b'def f():\n    return 1\n'
POST = b'def f():\n    return 2\n'
SRC = b'VALUE = 3\n'
BIN = b'\x7fELF binary'


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Apply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.repo, self.image, self.art, self.out = t / 'repo', t / 'image', t / 'art', t / 'out'
        work = t / 'work'
        (work / 'pkg').mkdir(parents=True)
        (work / 'pkg/mod.py').write_bytes(PRE)
        env = {'GIT_AUTHOR_NAME': 'x', 'GIT_AUTHOR_EMAIL': 'x', 'GIT_COMMITTER_NAME': 'x', 'GIT_COMMITTER_EMAIL': 'x',
               'HOME': str(t), 'PATH': '/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin'}
        subprocess.run(['git', 'init', '-q', str(work)], check=True, env=env)
        subprocess.run(['git', '-C', str(work), 'add', '.'], check=True, env=env)
        (work / 'pkg/mod.py').write_bytes(POST)
        patch = subprocess.run(['git', '-C', str(work), 'diff', '--full-index'], check=True, capture_output=True, env=env).stdout
        (self.repo / 'patches/x/tp4').mkdir(parents=True)
        (self.repo / 'patches/x/tp4/mod.py.patch').write_bytes(patch)
        (self.repo / 'src/tp4/ours').mkdir(parents=True)
        (self.repo / 'src/tp4/ours/new.py').write_bytes(SRC)
        (self.repo / 'sources').mkdir()
        manifest = {'schema': 2, 'install_root': DP, 'profiles': {'tp4': {'image': 'sha256:' + 'a' * 64, 'artefacts': [], 'files': [
            {'target': DP + 'pkg/mod.py', 'kind': 'patch', 'sha256': sha(POST), 'patch': 'patches/x/tp4/mod.py.patch',
             'preimage_sha256': sha(PRE)},
            {'target': DP + 'ours/new.py', 'kind': 'src', 'sha256': sha(SRC), 'source': 'src/tp4/ours/new.py'},
            {'target': '/opt/lib/thing.so', 'kind': 'binary', 'sha256': sha(BIN), 'artefact': 'thing'},
            {'target': '/opt/nccl', 'kind': 'binary-dir', 'artefact': 'nccl', 'files': {'lib.so.1': {'sha256': sha(BIN)}},
             'symlinks': {'lib.so': 'lib.so.1'}},
        ]}}}
        (self.repo / 'sources/installed-files.json').write_text(json.dumps(manifest))
        (self.image / DP.lstrip('/') / 'pkg').mkdir(parents=True)
        (self.image / DP.lstrip('/') / 'pkg/mod.py').write_bytes(PRE)
        self.art.mkdir()
        (self.art / sha(BIN)).write_bytes(BIN)

    def build(self, **kw):
        return apply.build('tp4', self.image, self.out, artefacts=self.art, root=self.repo, **kw)

    def test_builds_and_verifies_everything(self):
        receipt = self.build()
        self.assertEqual(receipt['problems'], [])
        self.assertEqual((self.out / DP.lstrip('/') / 'pkg/mod.py').read_bytes(), POST)
        self.assertEqual((self.out / DP.lstrip('/') / 'ours/new.py').read_bytes(), SRC)
        self.assertTrue((self.out / 'opt/nccl/lib.so').is_symlink())
        self.assertEqual(receipt['files']['/opt/lib/thing.so']['status'], 'measured-artefact')

    def test_wrong_image_file_is_refused(self):
        (self.image / DP.lstrip('/') / 'pkg/mod.py').write_bytes(PRE + b'# other image\n')
        receipt = self.build()
        self.assertTrue(any('not the recorded preimage' in p for p in receipt['problems']))
        self.assertFalse((self.out / DP.lstrip('/') / 'pkg/mod.py').exists())

    def test_tampered_patch_is_refused(self):
        p = self.repo / 'patches/x/tp4/mod.py.patch'
        p.write_bytes(p.read_bytes().replace(b'return 2', b'return 9'))
        receipt = self.build()
        self.assertTrue(any('differs from the served file' in p for p in receipt['problems']))

    def test_rebuilt_artefact_needs_explicit_permission(self):
        (self.art / sha(BIN)).unlink()
        (self.art / 'thing.so').write_bytes(BIN + b'rebuilt')
        (self.art / 'lib.so.1').write_bytes(BIN)
        receipt = self.build()
        self.assertTrue(any('differs from the measured bytes' in p for p in receipt['problems']))
        receipt = apply.build('tp4', self.image, Path(self.tmp.name) / 'out2', artefacts=self.art, root=self.repo,
                              allow_rebuilt=True)
        self.assertEqual(receipt['files']['/opt/lib/thing.so']['status'], 'rebuilt-artefact')

    def test_missing_artefact_is_reported(self):
        receipt = apply.build('tp4', self.image, self.out, artefacts=None, root=self.repo)
        self.assertEqual(receipt['files']['/opt/lib/thing.so']['status'], 'missing')

    def test_unsafe_target_is_refused(self):
        with self.assertRaises(ValueError):
            apply.safe_join(self.out, '/../../etc/passwd')


def metadata(name, version):
    return f'Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\nLong description.\nVersion: 9.9.9\n'.encode()


class HummingInstall(unittest.TestCase):
    """A humming-kernels 0.1.15 build installs `humming/` and `humming_kernels-0.1.15.dist-info/`; the profile mounts
    that dist-info at the image's `humming_kernels-0.1.12.dist-info` path (sources/installed-files.json, source_dir)."""
    INIT = b'__version__ = "0.1.15"\n'
    META = metadata('humming_kernels', '0.1.15')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        t = Path(self.tmp.name)
        self.repo, self.third, self.out = t / 'repo', t / 'site-packages', t / 'out'
        (self.repo / 'sources').mkdir(parents=True)
        self.manifest = {'schema': 2, 'install_root': DP, 'profiles': {'tp4': {'image': 'sha256:' + 'a' * 64, 'artefacts': [], 'files': [
            {'target': DP + 'humming', 'kind': 'third-party-dir', 'artefact': 'humming-kernels-0.1.15',
             'source_dir': 'humming', 'files': {'__init__.py': {'sha256': sha(self.INIT)}}},
            {'target': DP + 'humming_kernels-0.1.12.dist-info', 'kind': 'third-party-dir', 'artefact': 'humming-kernels-0.1.15',
             'source_dir': 'humming_kernels-0.1.15.dist-info', 'distribution': {'name': 'humming-kernels', 'version': '0.1.15'},
             'files': {'METADATA': {'sha256': sha(self.META)}}},
        ]}}}
        self.write_manifest()
        # what `pip install --target DIR` (or a venv's site-packages) of the 0.1.15 build contains
        (self.third / 'humming').mkdir(parents=True)
        (self.third / 'humming/__init__.py').write_bytes(self.INIT)
        (self.third / 'humming_kernels-0.1.15.dist-info').mkdir()
        (self.third / 'humming_kernels-0.1.15.dist-info/METADATA').write_bytes(self.META)

    def write_manifest(self):
        (self.repo / 'sources/installed-files.json').write_text(json.dumps(self.manifest))

    def build(self, out=None, **kw):
        return apply.build('tp4', Path(self.tmp.name) / 'image', out or self.out, third_party=self.third, root=self.repo, **kw)

    def test_build_output_is_mounted_at_the_image_path(self):
        receipt = self.build()
        self.assertEqual(receipt['problems'], [])
        target = self.out / DP.lstrip('/') / 'humming_kernels-0.1.12.dist-info/METADATA'
        self.assertEqual(target.read_bytes(), self.META)
        self.assertEqual(receipt['files'][DP + 'humming_kernels-0.1.12.dist-info/METADATA']['status'], 'measured-artefact')
        self.assertEqual((self.out / DP.lstrip('/') / 'humming/__init__.py').read_bytes(), self.INIT)

    def test_source_directory_is_required(self):
        del self.manifest['profiles']['tp4']['files'][1]['source_dir']
        self.write_manifest()
        receipt = self.build()
        self.assertTrue(any('source_dir' in p for p in receipt['problems']), receipt['problems'])
        self.assertFalse((self.out / DP.lstrip('/') / 'humming_kernels-0.1.12.dist-info/METADATA').exists())

    def test_other_version_is_refused_even_when_rebuilt_is_allowed(self):
        other = metadata('humming-kernels', '0.1.12')
        (self.third / 'humming_kernels-0.1.15.dist-info/METADATA').write_bytes(other)
        receipt = self.build(allow_rebuilt=True)
        self.assertTrue(any('expected humming-kernels 0.1.15' in p for p in receipt['problems']), receipt['problems'])
        self.assertFalse((self.out / DP.lstrip('/') / 'humming_kernels-0.1.12.dist-info/METADATA').exists())

    def test_rebuilt_dist_info_of_the_right_version_needs_permission(self):
        rebuilt = self.META + b'Requires-Dist: extra\n'
        (self.third / 'humming_kernels-0.1.15.dist-info/METADATA').write_bytes(rebuilt)
        receipt = self.build()
        self.assertTrue(any('differs from the measured bytes' in p for p in receipt['problems']))
        receipt = self.build(out=Path(self.tmp.name) / 'out2', allow_rebuilt=True)
        self.assertEqual(receipt['files'][DP + 'humming_kernels-0.1.12.dist-info/METADATA']['status'], 'rebuilt-artefact')

    def test_repository_manifest_names_every_source_directory(self):
        manifest = json.loads((ROOT / 'sources/installed-files.json').read_text())
        dirs = [i for prof in manifest['profiles'].values() for i in prof['files'] if i['kind'] == 'third-party-dir']
        self.assertTrue(dirs)
        for item in dirs:
            self.assertTrue(item.get('source_dir'), item['target'])
            if item['source_dir'].endswith('.dist-info'):
                dist = item['distribution']
                self.assertEqual(item['source_dir'], f"{dist['name'].replace('-', '_')}-{dist['version']}.dist-info")
                self.assertEqual(item['artefact'], f"{dist['name']}-{dist['version']}")


class RealPatches(unittest.TestCase):
    def test_every_repository_patch_parses_and_names_its_hashes(self):
        manifest = json.loads((ROOT / 'sources/installed-files.json').read_text())
        lock = json.loads((ROOT / 'sources.lock.json').read_text())
        locked = sum(len(lock['components'][c]['patches']) for c in ('vllm', 'b12x', 'sparse-mla'))
        n = 0
        for topo, prof in manifest['profiles'].items():
            for item in prof['files']:
                if item['kind'] != 'patch':
                    continue
                data = (ROOT / item['patch']).read_bytes()
                self.assertIn(f"Preimage:    sha256 {item['preimage_sha256']}".encode(), data)
                self.assertIn(f"Postimage:   sha256 {item['sha256']}".encode(), data)
                r = subprocess.run(['git', 'apply', '--numstat'], input=data, capture_output=True, cwd=ROOT)
                self.assertEqual(r.returncode, 0, item['patch'])
                n += 1
        self.assertEqual(n, locked)
        self.assertGreater(n, 30)


if __name__ == '__main__':
    unittest.main()
