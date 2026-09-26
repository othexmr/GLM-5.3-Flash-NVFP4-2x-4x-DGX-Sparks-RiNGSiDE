# SPDX-License-Identifier: Apache-2.0
"""docker/Dockerfile builds from pinned inputs only; docker/base holds the lab's stage inputs as recorded, and their
records lead to exactly the patch preimages of both profiles; the bake step serves every file of a profile."""
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / 'docker/Dockerfile'
BASE = ROOT / 'docker/base'
LAB_CHAIN = ['source', 's01-compat', 's02-kernels', 's03-flashkda', 's04-b12x-kda', 's05-layout', 's06-dense',
             's07-mtp', 's08-router', 's09-drafter', 's10-collectives', 's11-moe', 's12-reasoning',
             's13-deterministic-fastpath', 's14-attention-tail', 's15-topk-repair', 's16-deterministic-topk',
             's17-tool-call-truncation', 's19-runtime-bugfixes', 'v029-port', 's21-atomic-prefill', 'thinking-gate',
             'mixed-prefill-repair', 'base']
# Inputs resolved at build time on purpose; docs/build-provenance.md lists each of them.
FLOATING_PIP = {"'setuptools>=77.0.3,<81.0.0'"}


def module(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def instructions():
    """(stage, keyword, text) for every Dockerfile instruction; continuation lines joined, heredoc bodies attached."""
    lines = DOCKERFILE.read_text().splitlines()
    out, stage, i = [], None, 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        text = line
        while text.endswith('\\'):
            text = text[:-1] + ' ' + lines[i].strip()
            i += 1
        heredoc = re.search(r"<<'?(\w+)'?\s*$", text)
        if heredoc:
            body = []
            while lines[i] != heredoc.group(1):
                body.append(lines[i])
                i += 1
            i += 1
            text += '\n' + '\n'.join(body)
        keyword = text.split(None, 1)[0].upper()
        if keyword == 'FROM':
            m = re.match(r'FROM\s+(\S+)(?:\s+AS\s+(\S+))?\s*$', text, re.I)
            stage = m.group(2)
        out.append((stage, keyword, text))
    return out


def stages():
    """{stage: parent} from the FROM lines, and the global ARG defaults."""
    parents, args = {}, {}
    for stage, keyword, text in instructions():
        if keyword == 'FROM':
            ref = text.split()[1]
            parents[stage] = ref
        elif keyword == 'ARG' and stage is None:
            name, _, value = text.split(None, 1)[1].partition('=')
            args[name] = value
    return parents, args


def lock():
    return json.loads((ROOT / 'sources.lock.json').read_text())


class Dockerfile(unittest.TestCase):
    def test_every_base_is_a_stage_a_named_context_or_a_digest(self):
        parents, args = stages()
        seen = set()
        for stage, keyword, text in instructions():
            if keyword != 'FROM':
                continue
            ref = text.split()[1]
            if ref.startswith('${'):
                value = args[ref[2:-1]]
                self.assertRegex(value, r'@sha256:[0-9a-f]{64}$', text)
            else:
                self.assertTrue(ref in seen or ref in ('scratch', 'vllm-source'), text)
            seen.add(stage)

    def test_every_downloaded_input_is_pinned(self):
        for stage, keyword, text in instructions():
            if keyword == 'ADD' and '://' in text:
                self.assertRegex(text, r'--checksum=sha256:[0-9a-f]{64}\s', text)
            if keyword != 'RUN':
                continue
            for m in re.finditer(r'git fetch[^&]*?(https://\S+)\s+(\S+)', text):
                self.assertRegex(m.group(2), r'^[0-9a-f]{40}$', stage)
                self.assertRegex(text[m.end():], r"git rev-parse '?HEAD", stage)
            for m in re.finditer(r'(?:uv )?pip install ((?:[^&|]|&(?!&))*)', text):
                words = m.group(1).split()
                if '--no-index' in words or '-c' in words:
                    continue
                for w in words:
                    if w.startswith('-') or w.startswith('/') or w.startswith('"') or w in FLOATING_PIP:
                        continue
                    self.assertRegex(w, r'^[A-Za-z0-9_.\[\]-]+==[^=]+$', f'{stage}: unpinned {w}')
            self.assertNotRegex(text, r'curl[^|]*\|\s*(ba)?sh', stage)

    def test_package_managers_run_only_in_build_only_stages(self):
        parents, _ = stages()
        served = set()
        for top in ('profile-tp4', 'profile-tp2'):
            s = top
            while s in parents:
                served.add(s)
                s = parents[s]
        for stage, keyword, text in instructions():
            if keyword == 'RUN' and stage in served:
                self.assertNotRegex(text, r'\b(apt-get|dnf|yum) install', stage)

    def test_the_base_follows_the_lab_order(self):
        parents, _ = stages()
        for child, parent in zip(LAB_CHAIN[1:], LAB_CHAIN):
            self.assertEqual(parents[child], parent, child)
        self.assertEqual(parents['source'], 'source-upstream')
        self.assertEqual(parents['source-upstream'], 'vllm-source')
        for profile in ('profile-tp4', 'profile-tp2'):
            self.assertEqual(parents[profile], 'base')

    def test_referenced_inputs_exist(self):
        text = DOCKERFILE.read_text()
        for rel in set(re.findall(r'/inputs/([\w./-]+)', text)):
            self.assertTrue((BASE / rel).exists() or (BASE / (rel + '.b64')).exists(), rel)
        for rel in set(re.findall(r'--mount=type=bind,source=([\w./-]+),target', text)) - {'.'}:
            self.assertTrue((ROOT / rel).exists(), rel)
        for stage, keyword, line in instructions():
            if keyword == 'COPY' and '--from' not in line:
                self.assertTrue((ROOT / line.split()[1]).exists(), line)
        ignored = [p.strip().rstrip('/') for p in (ROOT / '.dockerignore').read_text().splitlines()
                   if p.strip() and not p.startswith('#')]
        for rel in re.findall(r'(?:source=|COPY )((?:docker|sources|patches|build|src)/[\w./-]*)', text):
            self.assertFalse(any(rel == p or rel.startswith(p + '/') for p in ignored), rel)

    def test_source_identities_equal_the_lock(self):
        text = DOCKERFILE.read_text()
        comps = lock()['components']
        for name in ('vllm', 'sparse-mla', 'nccl'):
            self.assertIn(comps[name]['base_commit'], text, name)
            self.assertIn(comps[name]['base_tree'], text, name)
        for patch in comps['nccl']['patches']:
            self.assertIn(patch['result_tree'], text)
        for patch in comps['sparse-mla']['build_patches']:
            self.assertIn(patch['result_tree'], text)
            self.assertIn(Path(patch['path']).name, text)
        self.assertIn('7ffd99abacd3e746', text)  # prefill RDMA library name = source sha256 prefix
        prb = hashlib.sha256((ROOT / 'src/tp4/glm53_prefill_rdma/_prb.cu').read_bytes()).hexdigest()
        self.assertTrue(prb.startswith('7ffd99abacd3e746'))


def sha256sums(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        digest, name = line.split('  ', 1)
        out[name.lstrip('*')] = digest
    return out


class BaseInputs(unittest.TestCase):
    def test_sums_cover_every_file(self):
        sums = sha256sums(BASE / 'SHA256SUMS')
        files = {p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()
                 and p.name != '.DS_Store' and '__pycache__' not in p.parts} - {'SHA256SUMS', 'README.md'}
        self.assertEqual(set(sums), files)
        for name, digest in sums.items():
            self.assertEqual(hashlib.sha256((BASE / name).read_bytes()).hexdigest(), digest, name)

    def test_base64_files_decode_to_the_lab_sums(self):
        stage = BASE / '04-b12x-kda'
        sums = sha256sums(stage / 'b12x-kda-prefill-overlay.SHA256SUMS')
        encoded = list(stage.rglob('*.json.gz.b64'))
        self.assertEqual(len(encoded), 2)
        for p in encoded:
            rel = p.relative_to(stage / 'b12x-kda-prefill-overlay').as_posix()[:-len('.b64')]
            self.assertEqual(hashlib.sha256(base64.b64decode(p.read_bytes())).hexdigest(), sums[rel], rel)
        overlay = stage / 'b12x-kda-prefill-overlay'
        present = {p.relative_to(overlay).as_posix().removesuffix('.b64') for p in overlay.rglob('*') if p.is_file()}
        self.assertEqual(present, set(sums))

    def test_source_distributions_carry_the_recorded_versions(self):
        dists = dict(line.split('==', 1) for line in (BASE / '00-source/distributions.txt').read_text().splitlines()
                     if line and not line.startswith('#'))
        for name, version in {'torch': '2.13.0+cu130', 'transformers': '5.16.1', 'flashinfer-python': '0.6.18',
                              'triton': '3.7.1', 'nvidia-cutlass-dsl': '4.6.2', 'tilelang': '0.1.12',
                              'apache-tvm-ffi': '0.1.11', 'humming-kernels': '0.1.12', 'instanttensor': '0.1.9',
                              'nvidia-nccl-cu13': '2.30.7'}.items():
            self.assertEqual(dists[name], version, name)
        self.assertIn('+g4500c80c', dists['vllm'])
        image = (ROOT / 'docs/image.md').read_text()
        for name in ('torch 2.13.0+cu130', 'transformers 5.16.1', 'flashinfer-python 0.6.18', 'triton 3.7.1'):
            self.assertIn(name, image)

    def test_stage_records_lead_to_every_patch_preimage(self):
        """The ported vLLM tree, then stage 21 and the mixed-prefill repair, give every vLLM preimage; stages 13 and
        19 the b12x preimages; stages 14, 19 and 21 the sparse-MLA backend preimage."""
        tree = {name[len('vllm/'):]: digest for name, digest in sha256sums(BASE / 'v029-port/vllm-tree.sha256').items()}
        self.assertEqual(len(tree), 2948)
        port = json.loads((BASE / 'v029-port/source-port-manifest.json').read_text())
        self.assertEqual(len(port['changes']), 118)
        for change in port['changes']:
            if change['path'].startswith('vllm/') and change['sha256']:
                self.assertEqual(tree[change['path'][5:]], change['sha256'], change['path'])
        state = {'vllm/' + k: v for k, v in tree.items()}

        def step(path, before, after):
            self.assertEqual(state.get(path), before, path)
            state[path] = after

        for row in json.loads((BASE / '21-atomic-prefill/source-manifest.json').read_text()):
            if row['path'].startswith('vllm/'):
                step(row['path'], row['before'], row['after'])
        mixed = json.loads((BASE / 'mixed-prefill-repair/overlay-receipt.json').read_text())
        for path, before in mixed['source'].items():
            step(path, before, mixed['overlay'][path])
        # b12x: stage 13, then stage 19; sparse-MLA backend: stage 14, 19, 21
        s13 = json.loads((BASE / '13-deterministic-fastpath/source-manifest.json').read_text())['files']
        s14 = json.loads((BASE / '14-attention-tail/source-manifest.json').read_text())['files']
        s19 = {r['path']: r for r in json.loads((BASE / '19-runtime-bugfixes/source-manifest.json').read_text())}
        s21 = {r['path']: r for r in json.loads((BASE / '21-atomic-prefill/source-manifest.json').read_text())}
        for path, row in s13.items():
            state[path] = row['after']
        state['glm53_sparse_mla/backend.py'] = s14['glm53_sparse_mla/backend.py']['before']
        step('glm53_sparse_mla/backend.py', s14['glm53_sparse_mla/backend.py']['before'],
             s14['glm53_sparse_mla/backend.py']['after'])
        for path in ('b12x/moe/fused_moe/_impl.py', 'glm53_sparse_mla/backend.py'):
            step(path, s19[path]['before'], s19[path]['after'])
        step('glm53_sparse_mla/backend.py', s21['glm53_sparse_mla/backend.py']['before'],
             s21['glm53_sparse_mla/backend.py']['after'])
        checked = 0
        for comp in ('vllm', 'b12x', 'sparse-mla'):
            for patch in lock()['components'][comp]['patches']:
                self.assertEqual(state.get(patch['target']), patch['preimage_sha256'], (comp, patch['target']))
                checked += 1
        self.assertEqual(checked, 54)
        # the thinking gate replaces the parser of the port
        gate = (BASE / 'thinking-gate/check_preimage.py').read_text()
        self.assertIn(tree['parser/glm53_moe.py'], gate)
        # stage 02's plugin backend patch result is stage 14's preimage
        self.assertIn(s14['glm53_sparse_mla/backend.py']['before'], DOCKERFILE.read_text())


class Bake(unittest.TestCase):
    def test_every_served_target_is_placed(self):
        bake = module('bake_overlay', 'docker/tools/bake_overlay.py')
        manifest = ROOT / 'sources/installed-files.json'
        for topo in ('tp4', 'tp2'):
            entry = json.loads(manifest.read_text())['profiles'][topo]
            with tempfile.TemporaryDirectory() as tmp:
                overlay, root = Path(tmp) / 'overlay', Path(tmp) / 'root'
                for item in entry['files']:
                    path = overlay / item['target'].lstrip('/')
                    if item['kind'] in bake.DIR_KINDS:
                        for name in item['files']:
                            (path / name).parent.mkdir(parents=True, exist_ok=True)
                            (path / name).write_text('new ' + name)
                        stale = root / item['target'].lstrip('/') / 'stale-from-image.py'
                        stale.parent.mkdir(parents=True, exist_ok=True)
                        stale.write_text('old')
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text('new')
                baked, problems = bake.bake(topo, overlay, manifest, root)
                self.assertEqual(problems, [])
                self.assertEqual(sorted(baked), sorted(i['target'] for i in entry['files']))
                for item in entry['files']:
                    target = root / item['target'].lstrip('/')
                    if item['kind'] in bake.DIR_KINDS:
                        self.assertEqual(sorted(p.relative_to(target).as_posix() for p in target.rglob('*')
                                                if p.is_file()), sorted(item['files']))
                    else:
                        self.assertEqual(target.read_text(), 'new')

    def test_check_served_names_differences(self):
        served = module('check_served', 'docker/tools/check_served.py')
        manifest = json.loads((ROOT / 'sources/installed-files.json').read_text())['profiles']['tp2']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for item in manifest['files']:
                path = root / item['target'].lstrip('/')
                path.parent.mkdir(parents=True, exist_ok=True)
                if item['kind'] == 'src' or item['kind'] == 'template':
                    path.write_bytes((ROOT / item['source']).read_bytes())
                elif item['kind'] == 'binary':
                    path.write_bytes(b'a rebuilt extension')
            rows = {r['path']: r['status'] for r in served.check('tp2', ROOT / 'sources/installed-files.json', root)}
        for item in manifest['files']:
            status = rows.get(item['target'])
            if item['kind'] in ('src', 'template'):
                self.assertEqual(status, 'measured', item['target'])
            elif item['kind'] == 'binary':
                self.assertEqual(status, 'rebuilt', item['target'])
            elif item['kind'] == 'patch':
                self.assertEqual(status, 'missing', item['target'])


class Provenance(unittest.TestCase):
    def test_docs_name_every_build_time_input(self):
        doc = (ROOT / 'docs/build-provenance.md').read_text()
        for phrase in ('setuptools>=77.0.3,<81.0.0', 'rustup', 'apt', 'dnf', 'ZJY0516/vllm', '4500c80c',
                       'e6b0dfc2b6fd307315e9b34b73cd2bfe7b6b08958eda721828e61732bba426b0',
                       '7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1',
                       'f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9', 'Blockers'):
            self.assertIn(phrase, doc)


if __name__ == '__main__':
    unittest.main()
