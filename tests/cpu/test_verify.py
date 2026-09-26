# SPDX-License-Identifier: Apache-2.0
"""Adversarial checks of sources/verify.py on a copy of the repository."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('verify', ROOT / 'sources/verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
COPY = ('sources.lock.json', 'LICENSE', 'NOTICE', 'sources', 'patches', 'src', 'launch', 'licenses', 'release', 'build')


class Verify(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'recipe'
        self.root.mkdir()
        for name in COPY:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, self.root / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copyfile(src, self.root / name)

    def cli(self, release=False):
        r = subprocess.run([sys.executable, '-B', str(ROOT / 'sources/verify.py'), '--root', str(self.root)]
                           + (['--release'] if release else []), capture_output=True, text=True)
        self.assertEqual(r.stderr, '')
        return r.returncode, json.loads(r.stdout)

    def invalid(self, pattern=None):
        code, body = self.cli()
        self.assertEqual((code, body['status']), (2, 'INVALID'), body)
        if pattern:
            self.assertIn(pattern, body['error'])

    def edit_json(self, relative, fn):
        path = self.root / relative
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data, indent=1))

    def lock_patch(self, component='vllm', profile='tp4'):
        lock = json.loads((self.root / 'sources.lock.json').read_text())
        return next(p for p in lock['components'][component]['patches'] if p['profile'] == profile)

    # --- baseline
    def test_repository_is_consistent_and_release_is_blocked(self):
        self.assertEqual(self.cli()[0], 0)
        code, body = self.cli(release=True)
        self.assertEqual((code, body['status']), (2, 'RELEASE_BLOCKED'))
        self.assertTrue(any('base image' in b for b in body['release_blockers']))
        self.assertFalse(body['gpu_or_serving_checked'])

    # --- patches
    def test_changed_patch_bytes_are_refused(self):
        entry = self.lock_patch()
        path = self.root / 'patches/vllm' / entry['path']
        path.write_bytes(path.read_bytes() + b'\n')
        self.invalid('patch hash differs')

    def test_patch_with_rehashed_but_wrong_blob_is_refused(self):
        entry = self.lock_patch()
        path = self.root / 'patches/vllm' / entry['path']
        data = path.read_bytes().replace(entry['postimage_blob'].encode(), b'0' * 39 + b'1', 1)
        path.write_bytes(data)
        self.edit_json('sources.lock.json', lambda d: next(p for p in d['components']['vllm']['patches']
                                                           if p['path'] == entry['path']).update(
            sha256=hashlib.sha256(data).hexdigest()))
        self.invalid('index blob ids differ')

    def test_patch_touching_another_file_is_refused(self):
        entry = self.lock_patch()
        self.edit_json('sources.lock.json', lambda d: next(p for p in d['components']['vllm']['patches']
                                                           if p['path'] == entry['path']).update(target='vllm/other.py'))
        self.invalid('touches other files')

    def test_series_must_match_the_lock(self):
        series = self.root / 'patches/vllm/series'
        lines = series.read_text().splitlines()
        body = [l for l in lines if not l.startswith('#')]
        series.write_text('\n'.join([l for l in lines if l.startswith('#')] + body[::-1]) + '\n')
        self.invalid('series file disagrees')

    def test_unlisted_patch_queue_file_is_refused(self):
        (self.root / 'patches/b12x/tp4/extra.patch').write_bytes(b'x')
        self.invalid('unlisted patch queue file')

    def test_patch_outside_its_profile_directory_is_refused(self):
        entry = self.lock_patch()
        self.edit_json('sources.lock.json', lambda d: next(p for p in d['components']['vllm']['patches']
                                                           if p['path'] == entry['path']).update(profile='tp2'))
        self.invalid()

    def test_symlinked_patch_is_refused(self):
        entry = self.lock_patch()
        path = self.root / 'patches/vllm' / entry['path']
        outside = Path(self.tmp.name) / 'outside.patch'
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.invalid('symlink')

    def test_nccl_series_final_tree_must_match(self):
        self.edit_json('sources.lock.json', lambda d: d['components']['nccl'].update(result_tree='a' * 40))
        self.invalid('final tree')

    # --- lock
    def test_upstream_must_be_canonical(self):
        self.edit_json('sources.lock.json', lambda d: d['components']['b12x'].update(upstream='https://example.org/b12x.git'))
        self.invalid('canonical public upstream')

    def test_floating_base_is_refused(self):
        self.edit_json('sources.lock.json', lambda d: d['components']['vllm'].update(base_commit='main'))
        self.invalid('invalid recorded base identity')

    def test_public_image_reference_must_be_pinned(self):
        self.edit_json('sources.lock.json', lambda d: d['base_image'].update(public_reference='registry.example/glm53:latest'))
        self.invalid('pinned by digest')

    def test_duplicate_keys_and_nan_are_refused(self):
        path = self.root / 'sources.lock.json'
        path.write_text('{"schema": 2, "schema": 2}')
        self.invalid('duplicate JSON key')
        path.write_text('{"schema": NaN}')
        self.invalid('non-JSON numeric constant')

    # --- manifest and profiles
    def test_changed_src_file_is_refused(self):
        path = self.root / 'src/tp4/glm53_speedup/policy.py'
        path.write_bytes(path.read_bytes() + b'# changed\n')
        self.invalid('differs from the served file')

    def test_changed_override_is_refused(self):
        (self.root / 'launch/profiles/tp2/override.json').write_text('{"mode": "off"}\n')
        self.invalid()

    def test_unmounted_served_file_is_refused(self):
        def drop(d):
            argv = d['argv_template']
            i = next(i for i, t in enumerate(argv) if t.startswith('type=bind,src=${OVERLAY_ROOT}') and 'policy.py' in t)
            del argv[i - 1:i + 1]
        self.edit_json('launch/profiles/tp4/profile.json', drop)
        self.invalid('not mounted')

    def test_mount_not_in_manifest_is_refused(self):
        self.edit_json('launch/profiles/tp2/profile.json', lambda d: d['argv_template'].extend(
            ['--mount', 'type=bind,src=${OVERLAY_ROOT}/opt/extra.py,dst=/opt/extra.py,readonly']))
        self.invalid('not in the manifest')

    def test_undeclared_placeholder_is_refused(self):
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: d['argv_template'].extend(['-e', 'X=${PRIVATE_THING}']))
        self.invalid('undeclared placeholder')

    def test_nondeterministic_profile_must_say_so(self):
        self.edit_json('launch/profiles/tp4/profile.json',
                       lambda d: d.update(deterministic=False, nondeterministic_note='fine'))
        self.invalid('must say so')

    def test_profile_image_must_match_the_lock(self):
        self.edit_json('launch/profiles/tp2/profile.json', lambda d: d['image'].update(id='sha256:' + 'b' * 64))
        self.invalid('image differs')

    # --- licences and assets
    def test_licence_texts_and_notice_are_required(self):
        (self.root / 'LICENSE').write_text('other')
        self.invalid('license text differs')

    def test_notice_must_credit_upstreams(self):
        path = self.root / 'NOTICE'
        path.write_text(path.read_text().replace('FujitsuPolycom/sparkring', 'someone'))
        self.invalid('NOTICE lacks')

    def test_every_artefact_needs_an_asset_entry(self):
        self.edit_json('release/assets.json', lambda d: d.update(artefacts=[a for a in d['artefacts'] if a['name'] != 'glm53-fp8-gemm']))
        self.invalid('not described')

    # --- launch identity: a flag, mount or served-file change needs a new plan import
    def set_kda_flag(self, data, value):
        argv = data['argv_template'] if 'argv_template' in data else None
        for tokens in ([argv] if argv is not None else data['argv'].values()):
            i = tokens.index('GLM53_KDA_CKPT=1')
            tokens[i] = 'GLM53_KDA_CKPT=' + value

    def test_changed_kda_flag_is_refused(self):
        plan = json.loads((self.root / 'launch/profiles/tp4/profile.json').read_text())['source_plan']['sha256']
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: self.set_kda_flag(d, '0'))
        self.invalid('lab plan ' + plan[:12])

    def test_changed_flag_with_an_edited_fixture_is_refused(self):
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: self.set_kda_flag(d, '0'))
        self.edit_json('launch/profiles/tp4/canonical-launch.json', lambda d: self.set_kda_flag(d, '0'))
        self.invalid('import a new plan')

    def test_rehashed_launch_must_be_recorded_everywhere(self):
        mod = importlib.util.spec_from_file_location('v2', ROOT / 'sources/verify.py')
        v = importlib.util.module_from_spec(mod)
        mod.loader.exec_module(v)
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: self.set_kda_flag(d, '0'))
        self.edit_json('launch/profiles/tp4/canonical-launch.json', lambda d: self.set_kda_flag(d, '0'))
        prof = json.loads((self.root / 'launch/profiles/tp4/profile.json').read_text())
        inst = json.loads((self.root / 'sources/installed-files.json').read_text())
        digest = v.launch_digest(prof, inst['profiles']['tp4'])
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: d['source_plan'].update(launch_sha256=digest))
        self.edit_json('launch/profiles/tp4/canonical-launch.json', lambda d: d.update(launch_sha256=digest))
        self.invalid('import a new plan')   # release/provenance.json still holds the lab plan's launch

    def test_changed_served_file_hash_changes_the_launch(self):
        def bump(d):
            item = next(i for i in d['profiles']['tp2']['files'] if i['kind'] == 'binary')
            item['sha256'] = 'e' * 64
        self.edit_json('sources/installed-files.json', bump)
        self.invalid()

    def test_qualification_must_be_the_derived_plans_own(self):
        prof = json.loads((self.root / 'launch/profiles/tp4/profile.json').read_text())
        if 'qualification' not in prof['source_plan']:
            self.skipTest('the TP4 profile is a measured plan')
        self.edit_json('launch/profiles/tp4/profile.json',
                       lambda d: d['source_plan']['qualification'].update(plan_sha256='c' * 64))
        self.invalid('qualification must be the derived plan')

    def test_qualification_needs_what_was_not_established(self):
        prof = json.loads((self.root / 'launch/profiles/tp4/profile.json').read_text())
        if 'qualification' not in prof['source_plan']:
            self.skipTest('the TP4 profile is a measured plan')
        self.edit_json('launch/profiles/tp4/profile.json',
                       lambda d: d['source_plan']['qualification'].pop('not_established'))
        self.invalid('not_established')

    def test_plan_identity_must_agree(self):
        self.edit_json('release/provenance.json', lambda d: d['profiles']['tp2'].update(lab_plan_sha256='d' * 64))
        self.invalid('lab plan identity differs')

    # --- mount representations
    def first_overlay(self, d):
        return next(k for k, v in d['mounts'].items() if v['source'].startswith('${OVERLAY_ROOT}'))

    def test_mount_table_must_equal_the_argv(self):
        self.edit_json('launch/profiles/tp4/profile.json', lambda d: d['mounts'][self.first_overlay(d)].update(readonly=False))
        self.invalid('mount table differs')

    def test_mount_option_with_another_key_is_refused(self):
        def extra(d):
            i = next(i for i, t in enumerate(d['argv_template']) if t.startswith('type=bind,src=${OVERLAY_ROOT}'))
            d['argv_template'][i] = d['argv_template'][i].replace(',readonly', ',bind-propagation=rshared,readonly')
        self.edit_json('launch/profiles/tp2/profile.json', extra)
        self.invalid('mount option must be')

    def test_writable_overlay_mount_is_refused(self):
        def writable(d):
            i = next(i for i, t in enumerate(d['argv_template']) if t.startswith('type=bind,src=${OVERLAY_ROOT}'))
            d['argv_template'][i] = d['argv_template'][i][:-len(',readonly')]
            dst = d['argv_template'][i].split('dst=', 1)[1]
            d['mounts'][dst]['readonly'] = False
        self.edit_json('launch/profiles/tp2/profile.json', writable)
        self.invalid('read-only')

    # --- digests
    def test_image_id_must_be_a_full_lowercase_digest(self):
        bad = 'sha256:' + 'A' * 64
        self.edit_json('sources.lock.json', lambda d: d['base_image'].update(id=bad))
        for topo in ('tp4', 'tp2'):
            self.edit_json(f'launch/profiles/{topo}/profile.json', lambda d: d['image'].update(id=bad))
        self.edit_json('sources/installed-files.json', lambda d: [p.update(image=bad) for p in d['profiles'].values()])
        self.edit_json('release/assets.json', lambda d: d['image'].update(id=bad))
        self.invalid('image ID must be')

    def test_public_reference_needs_a_full_digest(self):
        self.edit_json('sources.lock.json', lambda d: d['base_image'].update(public_reference='registry.example/glm53@sha256:abc'))
        self.invalid('pinned by digest')

    # --- one identity per artefact across the manifests
    def test_artefact_sha256_must_agree_across_manifests(self):
        self.edit_json('release/assets.json', lambda d: next(a for a in d['artefacts'] if a['name'] == 'sparse-mla-a1c').update(sha256='c' * 64))
        self.invalid('sparse-mla-a1c')

    def test_build_patch_artefact_must_agree(self):
        self.edit_json('sources.lock.json', lambda d: d['components']['sparse-mla']['build_patches'][0].update(artefact_sha256='c' * 64))
        self.invalid('build patch artefact sha256 differs')

    def test_per_rank_artefact_must_agree(self):
        self.edit_json('release/assets.json', lambda d: next(a for a in d['artefacts'] if a['name'] == 'glm53-prefill-rdma-libprb')['per_rank_sha256'].update({'3': 'c' * 64}))
        self.invalid('differs')

    def test_source_tree_must_agree(self):
        self.edit_json('release/provenance.json', lambda d: d['source_trees'].update(nccl='a' * 40))
        self.invalid('source tree differs')

    def test_unserved_asset_is_refused(self):
        self.edit_json('release/assets.json', lambda d: d['artefacts'].append(dict(d['artefacts'][0], name='stale-build')))
        self.invalid('no profile serves')

    def test_humming_mapping_must_agree(self):
        def other(d):
            item = next(i for i in d['profiles']['tp4']['files'] if i['target'].endswith('.dist-info'))
            item['source_dir'] = 'humming_kernels-0.1.12.dist-info'
        self.edit_json('sources/installed-files.json', other)
        self.invalid()


if __name__ == '__main__':
    unittest.main()
