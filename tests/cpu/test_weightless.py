# SPDX-License-Identifier: Apache-2.0
"""The optional weightless GLP-44 steering (launch/options/weightless/prepare.py): pinned inputs, the steered model.py
of the lab's verification window (78693488, one comment's author note left out: 8cbfea91), the default profiles unchanged, and none of msuiche/weightless's code in this repository.

The patcher of msuiche/weightless is read from $WEIGHTLESS_CACHE (a directory with the file, named by its file name or
sha256) or downloaded from GitHub at the pinned commit (WEIGHTLESS_OFFLINE=1: never downloaded); without either, the
tests that need it are skipped."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load('weightless_prepare', 'launch/options/weightless/prepare.py')
render = load('weightless_render', 'launch/render.py')
compose = load('weightless_compose', 'launch/compose.py')
_PATCHER = []

# Distinctive text of the patcher, hex-encoded so that this file does not contain it either: its marker, its GGUF
# reader and loader definitions, its source-block names and the first line of its forward block's projection.
MARKERS = [bytes.fromhex(h) for h in (
    '5b7374656572696e672d686f746669785d',
    '646566205f726561645f676775665f63766563',
    '646566205f6c6f61645f676775665f636f6e74726f6c5f766563746f72',
    '474755465f535243',
    '73746565725f73747265616d203d206c617965722e68635f706f7374',
)]
# The recipe's own guard loads the vector with the patcher's loader; that one call line equals a line of the
# patcher once indentation is stripped (the recipe's line is indented one level less).
OWN_CALL = 'raw = _load_gguf_control_vector(path)'


def patcher():
    """The pinned patcher bytes, or None (offline and no cache)."""
    if not _PATCHER:
        data = prepare.from_cache(os.environ.get('WEIGHTLESS_CACHE'), (prepare.UPSTREAM['sha256'],
                                  Path(prepare.UPSTREAM['path']).name), prepare.UPSTREAM['sha256'])
        if data is None and not os.environ.get('WEIGHTLESS_OFFLINE'):
            try:
                data = prepare.fetch(prepare.UPSTREAM_URL, prepare.UPSTREAM['sha256'], prepare.UPSTREAM['path'],
                                     timeout=20)
            except prepare.Refused:
                data = None
        _PATCHER.append(data)
    return _PATCHER[0]


def normalized(line):
    """A line without its indentation, and without the quotes, comma and newline escape of a string literal."""
    line = line.strip().rstrip(',').strip('\'"')
    return (line[:-2] if line.endswith('\\n') else line).strip()


def repository_files():
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if path.is_file() and not any(p in ('.git', '__pycache__') for p in rel.parts):
            yield rel, path


def canonical(topo):
    return json.loads((ROOT / 'launch/profiles' / topo / 'canonical-launch.json').read_text())


def profile(topo):
    return json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())


class PinnedConstants(unittest.TestCase):
    def test_launch_and_prepare_agree(self):
        self.assertEqual(render.WEIGHTLESS_MODEL_DST, prepare.MODEL_DST)
        self.assertEqual(render.WEIGHTLESS_VECTOR_DST, prepare.VECTOR_DST)
        self.assertEqual(render.WEIGHTLESS_ENV, tuple(f'{k}={v}' for k, v in prepare.ENV))
        self.assertEqual([k for k, _ in prepare.ENV], ['WEIGHTLESS_STEER_PATH', 'WEIGHTLESS_STEER_ALPHA',
                                                       'WEIGHTLESS_STEER_HOOK', 'WEIGHTLESS_VECTOR_SHA256'])
        self.assertEqual(dict(prepare.ENV)['WEIGHTLESS_VECTOR_SHA256'], prepare.VECTOR['sha256'])

    def test_served_file_is_the_profiles(self):
        manifest = json.loads((ROOT / 'sources/installed-files.json').read_text())
        (entry,) = [f for f in manifest['profiles']['tp4']['files'] if f['target'] == prepare.MODEL_DST]
        self.assertEqual((entry['sha256'], entry['patch'], entry['preimage_sha256']),
                         (prepare.SERVED_SHA256, prepare.SERVED_PATCH, prepare.PORT_SHA256))
        port = json.loads((ROOT / 'docker/base/v029-port/source-port-manifest.json').read_text())
        (change,) = [c for c in port['changes'] if c['path'] == prepare.MODEL_REL]
        self.assertEqual((change['original_sha256'], change['sha256']), (None, prepare.PORT_SHA256))

    def test_pins(self):
        self.assertRegex(prepare.UPSTREAM['commit'], r'^[0-9a-f]{40}$')
        self.assertRegex(prepare.VECTOR['revision'], r'^[0-9a-f]{40}$')
        self.assertIn(prepare.UPSTREAM['commit'], prepare.UPSTREAM_URL)
        self.assertTrue(prepare.UPSTREAM_URL.startswith('https://raw.githubusercontent.com/msuiche/weightless/'))
        self.assertTrue(prepare.VECTOR_URL.startswith('https://huggingface.co/' + prepare.VECTOR['repository']
                                                      + '/resolve/' + prepare.VECTOR['revision'] + '/'))
        self.assertEqual(prepare.STEERED_SHA256, '8cbfea91c45de7f2f1f61ca4164f36647583bb812a3eecb1901accf432c27db7')
        # the file the lab's verification window served, and the one comment line in which it differs
        self.assertEqual(prepare.TESTED_SHA256, '786934881d1c1f6b68ae71fda96b7a92818c39c13344dd2723632b0c0f5cd32e')
        self.assertEqual(prepare.TESTED_RELATION['line'], 755)
        self.assertTrue(prepare.EDITS[1]['add'][0].lstrip().startswith('# 2026-09-23 (GLP steering'))
        self.assertEqual(prepare.VECTOR['sha256'], '0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c')

    def test_edits_are_ordered_zero_context_hunks(self):
        last = 0
        for edit in prepare.EDITS:
            self.assertEqual(set(edit), {'at', 'remove', 'removed_sha256', 'add'})
            self.assertGreater(edit['at'], last)
            self.assertEqual(edit['removed_sha256'] is None, edit['remove'] == 0)
            self.assertTrue(all('\n' not in line for line in edit['add']))
            last = edit['at'] + edit['remove']

    def test_tokens_stay_out(self):
        request = prepare.make_request('https://huggingface.co/x', {'Authorization': 'Bearer not-a-token'})
        self.assertNotIn('Authorization', request.headers)
        self.assertEqual(request.unredirected_hdrs.get('Authorization'), 'Bearer not-a-token')
        handler = prepare.KeepAuthOnSameHost()
        for target, kept in (('https://huggingface.co/api/resolve-cache/x', True),
                             ('https://cas-bridge.example/signed', False)):
            new = handler.redirect_request(request, None, 302, 'Found', {}, target)
            self.assertEqual(new.unredirected_hdrs.get('Authorization') is not None, kept, target)
            self.assertNotIn('Authorization', new.headers)
        text = json.dumps(prepare.receipt(True))
        self.assertNotIn('Bearer', text)
        self.assertNotIn('token', text.lower())


class Build(unittest.TestCase):
    @unittest.skipUnless(shutil.which('git'), 'git not found')
    def test_served_model_rebuilds_from_this_repository(self):
        self.assertEqual(prepare.sha256(prepare.served_model()), prepare.SERVED_SHA256)

    @unittest.skipUnless(shutil.which('git'), 'git not found')
    def test_merge_reproduces_the_lab_file(self):
        data = patcher()
        if data is None:
            self.skipTest('the pinned patcher is neither in $WEIGHTLESS_CACHE nor downloadable')
        self.assertEqual(prepare.sha256(prepare.steered_model(data)), prepare.STEERED_SHA256)

    @unittest.skipUnless(shutil.which('git'), 'git not found')
    def test_prepare_writes_the_option_directory(self):
        data = patcher()
        if data is None:
            self.skipTest('the pinned patcher is neither in $WEIGHTLESS_CACHE nor downloadable')
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'cache'
            cache.mkdir()
            (cache / Path(prepare.UPSTREAM['path']).name).write_bytes(data)
            out = Path(tmp) / 'out'
            args = ['--out', str(out), '--cache', str(cache), '--offline', '--skip-vector']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(prepare.main(args), 0)
                self.assertEqual(prepare.sha256((out / 'model.py').read_bytes()), prepare.STEERED_SHA256)
                self.assertFalse((out / 'control.gguf').exists())
                self.assertEqual(json.loads((out / 'weightless-receipt.json').read_text())['model']['sha256'],
                                 prepare.STEERED_SHA256)
                (out / 'model.py').write_bytes(b'other bytes')
                self.assertEqual(prepare.main(args), 2)
                self.assertIn('exists with other content', err.getvalue())
                self.assertEqual(prepare.main(['--out', str(Path(tmp) / 'out2'), '--offline']), 2)
                self.assertIn('--offline', err.getvalue())
                self.assertFalse((Path(tmp) / 'out2').exists())

    def test_changed_inputs_are_refused(self):
        with self.assertRaises(prepare.Refused):
            prepare.upstream_replacements(b'PATCHES = ()\n')
        edit = dict(at=1, remove=1, removed_sha256=prepare.sha256(b'a\n'), add=('b',))
        self.assertEqual(prepare.apply_edits(b'a\nc\n', (edit,)), b'b\nc\n')
        with self.assertRaises(prepare.Refused):
            prepare.apply_edits(b'x\nc\n', (edit,))
        with self.assertRaises(prepare.Refused):
            prepare.expect(b'x', prepare.STEERED_SHA256, 'steered model.py')

    def test_literal_reader_does_not_run_code(self):
        source = 'import os\nA = "a"\nB = (A + "b",)\nC = os.system("false")\nD = f"{A}"\n'
        self.assertEqual(prepare.literal_strings(source), {'A': 'a', 'B': ('ab',)})


class Launch(unittest.TestCase):
    ON = {'SWITCHLESS_WEIGHTLESS_DIR': '/srv/switchless/weightless'}

    def test_default_profiles_are_unchanged(self):
        for topo in ('tp4', 'tp2'):
            fixture = canonical(topo)
            prof = profile(topo)
            self.assertIs(render.with_options(prof, fixture['site']), prof)
            for rank, argv in fixture['argv'].items():
                rendered = render.render(prof, fixture['site'], int(rank))
                self.assertEqual(rendered, argv)
                self.assertFalse(any('WEIGHTLESS' in t or 'weightless' in t for t in rendered))

    def test_option_adds_exactly_the_lab_variant_edits(self):
        """The lab's variant plan: the model.py mount from the steered file (in place), then just before the image the
        vector mount and the four variables; nothing else changes."""
        fixture, prof = canonical('tp4'), profile('tp4')
        site = dict(fixture['site'], **self.ON)
        for rank, base in fixture['argv'].items():
            argv = render.render(prof, site, int(rank))
            image = prof['image']['id']
            i = base.index(image)
            extra = ['--mount',
                     'type=bind,src=/srv/switchless/weightless/control.gguf,dst=/opt/weightless/control.gguf,readonly',
                     '-e', 'WEIGHTLESS_STEER_PATH=/opt/weightless/control.gguf',
                     '-e', 'WEIGHTLESS_STEER_ALPHA=2.0', '-e', 'WEIGHTLESS_STEER_HOOK=post_layer',
                     '-e', 'WEIGHTLESS_VECTOR_SHA256=' + prepare.VECTOR['sha256']]
            self.assertEqual(argv[i:i + len(extra)], extra)
            rest = argv[:i] + argv[i + len(extra):]
            diff = [(a, b) for a, b in zip(base, rest) if a != b]
            self.assertEqual(len(rest), len(base))
            self.assertEqual(diff, [(
                next(t for t in base if t.startswith('type=bind,') and f',dst={prepare.MODEL_DST},' in t),
                f'type=bind,src=/srv/switchless/weightless/model.py,dst={prepare.MODEL_DST},readonly')])

    def test_option_is_tp4_only(self):
        with self.assertRaises(ValueError):
            render.render(profile('tp2'), dict(canonical('tp2')['site'], **self.ON), 0)

    def test_compose_carries_the_option(self):
        fixture, prof = canonical('tp4'), profile('tp4')
        off_doc, off_env = compose.project(prof, fixture['site'], 0, compose.CONTEXT)
        on_doc, on_env = compose.project(prof, dict(fixture['site'], **self.ON), 0, compose.CONTEXT)
        off, on = off_doc['services']['glm53'], on_doc['services']['glm53']
        self.assertEqual([e for e in on['environment'] if e not in off['environment']], list(render.WEIGHTLESS_ENV))
        self.assertEqual([(v['source'], v['target'], v['read_only']) for v in on['volumes'] if v not in off['volumes']],
                         [('${SWITCHLESS_WEIGHTLESS_DIR}/model.py', prepare.MODEL_DST, True),
                          ('${SWITCHLESS_WEIGHTLESS_DIR}/control.gguf', prepare.VECTOR_DST, True)])
        self.assertEqual({k: v for k, v in on_env.items() if k not in off_env}, self.ON)
        self.assertEqual({k for k in set(on) | set(off) if on.get(k) != off.get(k)}, {'environment', 'volumes'})
        self.assertFalse('WEIGHTLESS' in json.dumps(off_doc) or 'WEIGHTLESS' in json.dumps(off_env))


class Licence(unittest.TestCase):
    def test_no_weightless_code_in_this_repository(self):
        findings = []
        for rel, path in repository_files():
            data = path.read_bytes()
            findings += [f'{rel}: marker {m.decode()!r}' for m in MARKERS if m in data]
        self.assertEqual(findings, [])

    def test_no_line_the_patcher_injects_is_in_this_repository(self):
        data = patcher()
        if data is None:
            self.skipTest('the pinned patcher is neither in $WEIGHTLESS_CACHE nor downloadable')
        patches, block = prepare.upstream_replacements(data)
        injected = set()
        for _, before, after in patches:
            injected |= set(after.splitlines()) - set(before.splitlines())
        injected |= set(block.splitlines())
        distinctive = {line.strip() for line in injected
                       if len(line.strip()) >= 30 and sum(c.isalpha() for c in line) >= 10} - {OWN_CALL}
        self.assertGreater(len(distinctive), 150)
        findings = []
        for rel, path in repository_files():
            try:
                text = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                continue
            lines = {normalized(line) for line in text.splitlines()}
            findings += [f'{rel}: {line[:60]!r}' for line in sorted(distinctive & lines)]
        self.assertEqual(findings, [])
        added = {line for edit in prepare.EDITS for line in edit['add']}
        self.assertEqual(added & injected, set())


class Docs(unittest.TestCase):
    def test_documented(self):
        config = (ROOT / 'docs/configuration.md').read_text()
        for key in [k for k, _ in prepare.ENV] + [render.WEIGHTLESS_SITE]:
            self.assertIn(f'| `{key}` |', config)
        self.assertIn(prepare.MARKER, (ROOT / 'docs/operations.md').read_text())
        self.assertIn('launch/options/weightless/prepare.py', (ROOT / 'README.md').read_text())
        for rel in ('docs/credits.md', 'NOTICE'):
            text = (ROOT / rel).read_text()
            self.assertIn('msuiche/weightless', text)
            self.assertIn(prepare.VECTOR['repository'], text)


if __name__ == '__main__':
    unittest.main()
