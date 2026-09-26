# SPDX-License-Identifier: Apache-2.0
"""Import one lab launch plan into this repository (maintainer tool; never contacts a node).

Input: an import document (schema "switchless-import/1", see sources/README.md) produced from the lab's launch
plan, staging receipts and local copies of every served file. It carries private site values; this tool uses them
only to normalise the command and to render-check the result, and refuses to write any of them.

Output, for the named topology only (tp4 or tp2), replacing that profile's previous files:
  launch/profiles/<tp>/profile.json   argv template with ${...} placeholders, mount table, identities
  launch/profiles/<tp>/override.json  exact runtime override bytes
  launch/profiles/<tp>/chat_template_mm.jinja
  launch/profiles/<tp>/canonical-launch.json   the rendered commands for a documentation site, with the launch digest
  patches/<component>/<tp>/*.patch     one single-file patch per served upstream file, against the image's file
  src/<tp>/...                         full served bytes of files that are not upstream-owned image files
  sources/installed-files.json         the profile's served file set (sha256 of every served byte)
  sources.lock.json                    patch entries of the profile (other profiles are kept)
  release/provenance.json              the profile's lab plan and launch digest (other fields are kept)

Every patch is verified here: applied with `git apply` (no fuzz, no three-way) to the preimage, the result must
have the served sha256. All outputs are first written to a staging directory; every staged file is scanned for the
document's private values and for private paths and addresses, and every copy is checked against its served sha256.
Only then are the profile's directories and the shared files swapped in; on any failure the repository is left as it
was.

Usage: python3 -B sources/import_plan.py IMPORT.json [--root .]
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

DP = '/usr/local/lib/python3.12/dist-packages/'
PLACEHOLDER = re.compile(r'\$\{[A-Z0-9_:./-]+\}')
COMPONENT_DIRS = {'vllm': 'vllm', 'b12x': 'b12x', 'sparse-mla': 'sparse-mla'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_blob(data):
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def slug(rel):
    parts = rel.split('/')
    if parts[0] in ('vllm', 'b12x', 'glm53_sparse_mla'):
        parts = parts[1:]
    return '-'.join(parts) + '.patch'


def make_patch(rel, pre, post):
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                   GIT_AUTHOR_NAME='x', GIT_AUTHOR_EMAIL='x', GIT_COMMITTER_NAME='x', GIT_COMMITTER_EMAIL='x')
        subprocess.run(['git', 'init', '-q', tmp], check=True, env=env)
        target = Path(tmp) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pre)
        subprocess.run(['git', '-C', tmp, 'add', rel], check=True, env=env)
        target.write_bytes(post)
        diff = subprocess.run(['git', '-C', tmp, 'diff', '--full-index', '--no-color', '--no-ext-diff', '--', rel],
                              check=True, capture_output=True, env=env).stdout
    if not diff:
        raise SystemExit('served file equals its preimage: ' + rel)
    return diff


def check_patch(rel, pre, patch, want):
    """Apply exactly as sources/apply.py does and compare the result with the served sha256."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pre)
        for extra in (['--check'], []):
            subprocess.run(['git', 'apply', '--unsafe-paths', '--whitespace=nowarn', '-C', '3'] + extra,
                           input=patch, cwd=tmp, check=True, capture_output=True)
        got = sha256(target.read_bytes())
    if got != want:
        raise SystemExit(f'patch for {rel} does not reproduce the served file: {got} != {want}')


def header(topo, entry, rel, notices):
    pre = entry['preimage']
    lines = [
        f'RiNGSiDE overlay patch, profile {topo}',
        f'Target:      {entry["target"]}',
        f'Preimage:    sha256 {pre["sha256"]}',
        f'             {pre["origin"]}',
        f'Upstream:    {pre["upstream_relation"]}',
        f'Evidence:    {pre["evidence"]}',
        f'Postimage:   sha256 {entry["sha256"]} (the file the {topo} profile serves)',
        f'Licence:     {entry["licence"]}',
    ]
    if entry.get('attribution'):
        lines.append('Attribution: ' + ', '.join(entry['attribution']) + ' (see NOTICE)')
    lines += [f'Apply:       sources/apply.py (git apply, no fuzz, no three-way; both sha256 checked)', '---', '']
    return '\n'.join(lines).encode()


def canonical(argv):
    """Docker mount order is not significant here (no mount is nested in another): compare mounts as a multiset."""
    rest, mounts, i = [], [], 0
    while i < len(argv):
        if argv[i] == '--mount':
            mounts.append(argv[i + 1])
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    return rest, sorted(mounts)


def normalise(doc):
    """Return the rank-0 argv template and the per-rank render values; checks every rank's argv."""
    site = doc['site']
    envmap, argmap, mounts = site['env'], site['args'], site['mount_sources']
    overlay = {e['target'] for e in doc['files']} | {d['target'] for d in doc['dirs']} | {b['target'] for b in doc['binaries']}

    def template(cmd, rank):
        out, i = [], 0
        head_args = None
        while i < len(cmd):
            a = cmd[i]
            if a == '--name':
                out += [a, '${SWITCHLESS_CONTAINER_NAME}']
                i += 2
                continue
            if a == '--mount':
                parts = cmd[i + 1].split(',')
                kv = dict(p.split('=', 1) for p in parts if '=' in p)
                dst = kv['dst']
                if dst in mounts:
                    src = '${' + mounts[dst] + '}'
                elif dst in overlay:
                    src = '${OVERLAY_ROOT}' + dst
                else:
                    raise SystemExit('unmapped mount ' + dst)
                out += [a, ','.join(('src=' + src) if p.startswith('src=') else p for p in parts)]
                i += 2
                continue
            if a in ('-e', '--env'):
                k, _, v = cmd[i + 1].partition('=')
                if k in envmap:
                    v = '${' + envmap[k] + '}'
                out += [a, k + '=' + v]
                i += 2
                continue
            if a == '--node-rank':
                out += [a, '${RANK}']
                i += 2
                continue
            if a == '--host' and cmd[i + 2:i + 3] == ['--port']:
                head_args = [a, cmd[i + 1], '--port', '${' + argmap['--port'] + '}']
                out.append('${ROLE_ARGS}')
                i += 4
                continue
            if a == '--headless':
                out.append('${ROLE_ARGS}')
                i += 1
                continue
            if a in argmap and a != '--port':
                out += [a, '${' + argmap[a] + '}']
                i += 2
                continue
            out.append(a)
            i += 1
        return out, head_args

    tmpl0, head_args = template(doc['argv']['0'], '0')
    for r, cmd in doc['argv'].items():
        t, _ = template(cmd, r)
        if canonical(t) != canonical(tmpl0):
            raise SystemExit(f'rank {r} differs from rank 0 beyond the declared site values')
    for tok in tmpl0:
        if '${' in tok and not PLACEHOLDER.search(tok):
            raise SystemExit('malformed placeholder in ' + tok)
    return tmpl0, head_args


def render(tmpl, head_args, values, rank, mount_src):
    """Mirror of launch/render.py for the render check."""
    out = []
    expanded = []
    for tok in tmpl:
        expanded += (head_args if rank == '0' else ['--headless']) if tok == '${ROLE_ARGS}' else [tok]
    for tok in expanded:

        def sub(m):
            key = m.group(0)[2:-1]
            if key == 'RANK':
                return rank
            if key == 'OVERLAY_ROOT':
                return '${OVERLAY_ROOT}'
            return values[key]
        out.append(PLACEHOLDER.sub(sub, tok))
    # overlay mount sources are compared by content, not by path
    fixed = []
    for i, tok in enumerate(out):
        if i and out[i - 1] == '--mount' and 'src=${OVERLAY_ROOT}' in tok:
            dst = dict(p.split('=', 1) for p in tok.split(',') if '=' in p)['dst']
            tok = tok.replace('src=${OVERLAY_ROOT}' + dst, 'src=' + mount_src[dst])
        fixed.append(tok)
    return fixed


OVERLAY_DOCS = {
    'vllm': ('vLLM', '98dff2a81d747d1dba01a47f939f48c3526d4206 (v0.29.0)',
             'The base image carries vLLM v0.29.0 with the GLM-5.3 backport; the "upstream relation" column says whether '
             'the image file equals the v0.29.0 file, is a backport file, or already carries an earlier lab change.'),
    'b12x': ('b12x', '3a437ab5168060e4d625f05e1625c04089f1ba37 (b12x 1.3.0 for these files)',
             'The image files already carry earlier lab MoE changes on top of b12x 1.3.0.'),
    'sparse-mla': ('Sparse-MLA plugin', 'b1f20638d7f82f880e4bcb2136ee2f19c06f38a4',
                   '`build/` holds two patches against the plugin repository itself, from which the compiled extensions '
                   'were built (`build/sparse-mla/README.md`); they are not overlay patches and are not in `series`.'),
}


def write_patch_docs(out, lock):
    """Regenerate the README and provenance.json of the overlay patch queues from the lock (below `out`)."""
    for comp, (name, base, note) in OVERLAY_DOCS.items():
        c = lock['components'][comp]
        url = c['upstream'][:-len('.git')]
        rows = ['| profile | patch | target | preimage (image file) | upstream relation |', '|---|---|---|---|---|']
        rows += [f"| {e['profile']} | `{e['path']}` | `{e['target']}` | `{e['preimage_sha256'][:12]}` | {e['preimage_upstream']} |"
                 for e in c['patches']]
        text = (f"# {name} patches\n\nUpstream: [{url.replace('https://github.com/', '')}]({url}), reference base\n`{base}`.\n\n"
                'Single-file overlay patches against files of the base image (`docs/image.md`), one directory per profile. '
                'Each patch\nstarts with a header naming its target, the image file\'s sha256 (preimage), the served file\'s '
                'sha256 (postimage), the\nlicence and any attribution, followed by a Git diff with full blob ids. '
                '`sources/apply.py` applies them with\n`git apply` (no fuzz, no three-way merge) and checks both hashes. '
                'Patches are independent of each other; `series`\nlists them in a fixed order.\n\n'
                f'{note}\n\n' + '\n'.join(rows) + '\n')
        folder = out / 'patches' / comp
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'README.md').write_text(text)
        prov = {'schema': 2, 'kind': 'file-overlay', 'upstream': c['upstream'], 'reference_base': base,
                'authors': 'the GLM-5.3 RiNGSiDE recipe lab (othexmr), except the third-party changes named in each '
                           'patch header and in NOTICE',
                'licence': 'upstream licence of each file (Apache-2.0); see licenses/README.md',
                'patches': [{'path': e['path'], 'profile': e['profile'], 'target': e['target']} for e in c['patches']]}
        (folder / 'provenance.json').write_text(json.dumps(prov, indent=1) + '\n')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Documentation values only (RFC 5737 addresses, /srv paths): the site of the canonical launch fixture.
CANONICAL_SITE = {
    'SWITCHLESS_OVERLAY_ROOT': '/srv/switchless/overlay',
    'SWITCHLESS_MODEL_DIR': '/srv/models/nvidia-GLM-5.3-Flash-NVFP4',
    'SWITCHLESS_TOKENIZER_FILE': '/srv/models/nvidia-GLM-5.3-Flash-NVFP4/tokenizer.json',   # a profile that mounts one
    'SWITCHLESS_DRAFTER_REPO_DIR': '/srv/hf-cache/hub/models--incoai--GLM-5.3-Flash-DFlash2',
    'SWITCHLESS_HF_CACHE_DIR': '/srv/switchless/cache',
    'SWITCHLESS_JIT_CACHE_ROOT': '/srv/switchless/jit',
    'SWITCHLESS_PROFILES_ROOT': '/srv/switchless/profiles',
    'SWITCHLESS_CONTAINER_PREFIX': 'switchless',
    'SWITCHLESS_MASTER_ADDR': '192.0.2.10',
    'SWITCHLESS_MASTER_PORT': '25104',
    'SWITCHLESS_API_PORT': '8888',
    'SWITCHLESS_SOCKET_IFNAME': 'eth0',
}
CANONICAL_FABRIC = {
    'tp4': {'SWITCHLESS_RANK_HOST_IPS': '192.0.2.10,192.0.2.11,192.0.2.12,192.0.2.13',
            'SWITCHLESS_IB_HCA': '=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1',
            'SWITCHLESS_FABRIC_CIDR': '10.100.224.0/22'},
    'tp2': {'SWITCHLESS_RANK_HOST_IPS': '192.0.2.10,192.0.2.11',
            'SWITCHLESS_IB_HCA': 'rocep1s0f1,roceP2p1s0f1',
            'SWITCHLESS_FABRIC_CIDR': '10.100.224.0/24'},
}
# Generic private patterns refused in every output, next to the import document's own site values.
PRIVATE_PATTERNS = [re.compile(rb'/(?:Users)/|/(?:home)/[a-z]'),  # written so that this file does not match itself
                    re.compile(rb'\b(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b')]


def build(doc, root, out):
    """Write every output of the import below `out` (a staging tree mirroring repository paths); touch nothing else.
    Returns (owned directories replaced as a whole, shared files replaced, summary)."""
    topo = doc['topology']
    ranks = sorted(doc['argv'], key=int)

    # 1. argv template and render check against every rank of the measured plan
    tmpl, head_args = normalise(doc)
    values = dict(doc['site']['values'])
    for r in ranks:
        cmd = doc['argv'][r]
        mount_src = {}
        for i, tok in enumerate(cmd):
            if i and cmd[i - 1] == '--mount':
                kv = dict(p.split('=', 1) for p in tok.split(',') if '=' in p)
                mount_src[kv['dst']] = kv['src']
        v = dict(values, **doc['site']['per_rank'][r])
        if canonical(render(tmpl, head_args, v, r, mount_src)) != canonical(cmd):
            raise SystemExit(f'render check failed for rank {r}')

    owned = [f'launch/profiles/{topo}', f'src/{topo}'] + [f'patches/{c}/{topo}' for c in COMPONENT_DIRS.values()]
    prof = out / 'launch/profiles' / topo
    prof.mkdir(parents=True, exist_ok=True)

    installed, lock_entries = [], {c: [] for c in COMPONENT_DIRS}
    for e in sorted(doc['files'], key=lambda x: x['target']):
        data = Path(e['local']).read_bytes()
        if sha256(data) != e['sha256']:
            raise SystemExit('local copy differs from the served sha256: ' + e['target'])
        rel = e['target'][len(DP):] if e['target'].startswith(DP) else e['target'].lstrip('/')
        item = {'target': e['target'], 'sha256': e['sha256'], 'bytes': len(data), 'kind': e['kind'],
                'licence': e['licence'], 'attribution': e.get('attribution', []), 'evidence': e.get('evidence')}
        if e['kind'] == 'patch':
            pre = Path(e['preimage']['local']).read_bytes()
            if sha256(pre) != e['preimage']['sha256']:
                raise SystemExit('preimage copy differs: ' + rel)
            diff = make_patch(rel, pre, data)
            patch = header(topo, e, rel, None) + diff
            check_patch(rel, pre, patch, e['sha256'])
            comp = COMPONENT_DIRS[e['component']]
            path = out / 'patches' / comp / topo / slug(rel)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(patch)
            relpatch = f'{topo}/{slug(rel)}'
            item.update(patch=f'patches/{comp}/{relpatch}', preimage_sha256=e['preimage']['sha256'],
                        preimage_evidence=e['preimage']['evidence'])
            lock_entries[e['component']].append({
                'path': relpatch, 'sha256': sha256(patch), 'target': rel, 'profile': topo,
                'preimage_sha256': e['preimage']['sha256'], 'preimage_blob': git_blob(pre),
                'preimage_upstream': e['preimage']['upstream_relation'],
                'postimage_sha256': e['sha256'], 'postimage_blob': git_blob(data)})
        elif e['kind'] == 'src':
            dest = out / 'src' / topo / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            item['source'] = f'src/{topo}/{rel}'
            for k in ('lab_served_sha256', 'recipe_change'):   # a file whose SPDX line changed with the repository licence
                if e.get(k):
                    item[k] = e[k]
            if e.get('replaces_image_sha256'):
                item['replaces_image_sha256'] = e['replaces_image_sha256']
        elif e['kind'] == 'template':
            dest = prof / Path(rel).name
            dest.write_bytes(data)
            item['source'] = f'launch/profiles/{topo}/{Path(rel).name}'
            item['derived_from_sha256'] = e['preimage']['sha256']
        else:
            raise SystemExit('unknown kind ' + e['kind'])
        installed.append(item)

    artefacts = []
    for d in sorted(doc['dirs'], key=lambda x: x['target']):
        item = {'target': d['target'], 'kind': d['kind'], 'files': {}, 'licence': d['licence'],
                'attribution': d.get('attribution', [])}
        if d['kind'] == 'config-dir':
            f = d['files']['override.json']
            data = Path(f['local']).read_bytes()
            if sha256(data) != f['sha256']:
                raise SystemExit('override copy differs from the served sha256')
            (prof / 'override.json').write_bytes(data)
            item['files']['override.json'] = {'sha256': f['sha256'], 'source': f'launch/profiles/{topo}/override.json'}
        elif d['kind'] == 'src-dir':
            rel = d['target'][len(DP):]
            for name, f in sorted(d['files'].items()):
                if f.get('binary'):
                    item['files'][name] = {'sha256': f['sha256'], 'artefact': f['artefact']}
                    if f.get('per_rank_sha256'):
                        item['files'][name]['per_rank_sha256'] = f['per_rank_sha256']
                    artefacts.append({'name': f['artefact'], 'target': d['target'] + '/' + name, 'sha256': f['sha256'],
                                      'per_rank_sha256': f.get('per_rank_sha256')})
                    continue
                data = Path(f['local']).read_bytes()
                if sha256(data) != f['sha256']:
                    raise SystemExit('local copy differs from the served sha256: ' + d['target'] + '/' + name)
                dest = out / 'src' / topo / rel / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                item['files'][name] = {'sha256': f['sha256'], 'source': f'src/{topo}/{rel}/{name}'}
        elif d['kind'] in ('binary-dir', 'third-party-dir'):
            item['artefact'] = d['artefact']
            item['files'] = {k: {'sha256': v['sha256']} for k, v in sorted(d['files'].items())}
            if d.get('symlinks'):
                item['symlinks'] = d['symlinks']
            if d.get('ignored_pycache'):
                item['ignored_pycache_files'] = d['ignored_pycache']
            if d['kind'] == 'third-party-dir':
                if not d.get('source_dir') or '/' in d['source_dir']:
                    raise SystemExit('third-party directory without a source_dir: ' + d['target'])
                item['source_dir'] = d['source_dir']
                if d.get('distribution'):
                    item['distribution'] = d['distribution']
            artefacts.append({'name': d['artefact'], 'target': d['target']})
        else:
            raise SystemExit('unknown dir kind ' + d['kind'])
        installed.append(item)
    for b in sorted(doc['binaries'], key=lambda x: x['target']):
        installed.append({'target': b['target'], 'kind': 'binary', 'sha256': b['sha256'], 'bytes': b.get('bytes'),
                          'artefact': b['artefact'], 'evidence': b.get('evidence')})
        artefacts.append({'name': b['artefact'], 'target': b['target'], 'sha256': b['sha256']})

    # 2. manifests shared with the other profile (read from the repository, written to the staging tree)
    inst_path = root / 'sources/installed-files.json'
    inst = json.loads(inst_path.read_text()) if inst_path.exists() else {}
    if inst.get('schema') != 2:
        inst = {'schema': 2, 'install_root': DP, 'profiles': {}}
    inst['profiles'][topo] = {'image': doc['image']['id'], 'files': installed, 'artefacts': artefacts}
    inst['profiles'] = dict(sorted(inst['profiles'].items()))
    (out / 'sources').mkdir(parents=True, exist_ok=True)
    (out / 'sources/installed-files.json').write_text(json.dumps(inst, indent=1) + '\n')

    lock = json.loads((root / 'sources.lock.json').read_text())
    shared = ['sources/installed-files.json', 'sources.lock.json']
    for comp, entries in lock_entries.items():
        c = lock['components'][comp]
        c['patches'] = [p for p in c.get('patches', []) if p.get('profile') != topo] + entries
        c['patches'].sort(key=lambda p: (p['profile'], p['path']))
        folder = out / 'patches' / COMPONENT_DIRS[comp]
        folder.mkdir(parents=True, exist_ok=True)
        lines = ['# Single-file overlay patches against files of the base image, one directory per profile.',
                 '# Each patch is independent; sources/apply.py applies them and checks pre- and postimage sha256.']
        (folder / 'series').write_text('\n'.join(lines + [p['path'] for p in c['patches']]) + '\n')
        shared += [f'patches/{COMPONENT_DIRS[comp]}/{n}' for n in ('series', 'README.md', 'provenance.json')]
    (out / 'sources.lock.json').write_text(json.dumps(lock, indent=1) + '\n')
    write_patch_docs(out, lock)

    # 3. profile, its launch identity and the canonical launch fixture
    mounts = {}
    for i, tok in enumerate(tmpl):
        if i and tmpl[i - 1] == '--mount':
            kv = dict(p.split('=', 1) for p in tok.split(',') if '=' in p)
            mounts[kv['dst']] = {'source': kv['src'], 'readonly': tok.endswith(',readonly')}
    site_vars = sorted({m.group(0)[2:-1] for tok in tmpl + head_args for m in PLACEHOLDER.finditer(tok)}
                       - {'RANK', 'ROLE_ARGS', 'OVERLAY_ROOT'} | {'SWITCHLESS_CONTAINER_NAME'})
    profile = {
        'schema': 2,
        'topology': topo,
        'ranks': len(ranks),
        'status': 'derived-from-measured-lab-plan' if doc['plan'].get('derived_from') else 'measured-lab-plan',
        'deterministic': doc['plan'].get('deterministic'),  # false, or null when the measured plan does not assert it
        'nondeterministic_note': doc['plan'].get('nondeterministic_note'),
        'source_plan': {'sha256': doc['plan']['sha256'], 'label': doc['plan'].get('label'),
                        **({'derived_from': doc['plan']['derived_from']} if doc['plan'].get('derived_from') else {}),
                        # the evidence for a derived plan, kept apart from its derivation (the document's own record)
                        **({'qualification': doc['plan']['qualification']} if doc['plan'].get('qualification') else {}),
                        'render_check': 'every rank of the lab plan re-rendered from this template with the lab site '
                                        'values: all tokens equal, mounts equal as a set (their order differs '
                                        'between the ranks and is not significant), overlay mount sources matched '
                                        'by sha256'},
        'image': {'id': doc['image']['id'], 'tag': doc['image']['tag'], 'public_reference': None},
        'argv_template': tmpl,
        'role_args': {'head': head_args, 'worker': ['--headless']},
        'mounts': mounts,
        'site_variables': site_vars,
        'overlay': f'sources/installed-files.json#{topo}',
    }
    here = Path(__file__).resolve().parents[1]
    verify = load_module('switchless_verify', here / 'sources/verify.py')
    renderer = load_module('switchless_render', here / 'launch/render.py')
    digest = verify.launch_digest(profile, inst['profiles'][topo])
    profile['source_plan']['launch_sha256'] = digest
    (prof / 'profile.json').write_text(json.dumps(profile, indent=1) + '\n')
    site = dict(CANONICAL_SITE, **CANONICAL_FABRIC[topo])
    fixture = {'schema': 1, 'topology': topo, 'source_plan_sha256': doc['plan']['sha256'], 'launch_sha256': digest,
               'note': 'launch/render.py output for the documentation site below; sources/verify.py re-renders it '
                       'and recomputes launch_sha256, so any change of a flag, mount or served file needs a new import',
               'site': site,
               'argv': {str(r): renderer.render(profile, site, r) for r in range(len(ranks))}}
    (prof / 'canonical-launch.json').write_text(json.dumps(fixture, indent=1) + '\n')

    prov_path = root / 'release/provenance.json'
    if prov_path.exists():
        prov_doc = json.loads(prov_path.read_text())
        entry = prov_doc.setdefault('profiles', {}).setdefault(topo, {})
        entry['lab_plan_sha256'] = doc['plan']['sha256']
        entry['launch_sha256'] = digest
        (out / 'release').mkdir(parents=True, exist_ok=True)
        (out / 'release/provenance.json').write_text(json.dumps(prov_doc, indent=1) + '\n')
        shared.append('release/provenance.json')

    summary = {'topology': topo, 'plan': doc['plan']['sha256'][:12], 'launch': digest[:12],
               'patches': sum(len(v) for v in lock_entries.values()),
               'src_files': sum(1 for i in installed if i.get('source', '').startswith('src/')),
               'artefacts': len(artefacts), 'render_check': 'ok', 'ranks': len(ranks)}
    return owned, shared, summary


def validate(doc, out, owned, shared):
    """Before anything is written to the repository: every output file is scanned for private values, and every copy
    is checked against the served sha256 recorded in the staged manifest."""
    values = doc['site']['values']
    private = {values[k] for k in ('SWITCHLESS_MASTER_ADDR', 'SWITCHLESS_MODEL_DIR', 'SWITCHLESS_DRAFTER_REPO_DIR',
                                   'SWITCHLESS_HF_CACHE_DIR', 'SWITCHLESS_TOKENIZER_FILE') if values.get(k)}
    private |= {v for pr in doc['site']['per_rank'].values() for v in pr.values() if v}
    private = sorted(v.encode() for v in private)
    files = [p for d in owned if (out / d).exists() for p in sorted((out / d).rglob('*')) if p.is_file()]
    files += [out / f for f in shared]
    for p in files:
        data = p.read_bytes()
        rel = p.relative_to(out).as_posix()
        for v in private:
            if v in data:
                raise SystemExit(f'private site value would reach the repository: {rel}')
        for pattern in PRIVATE_PATTERNS:
            m = pattern.search(data)
            if m:
                raise SystemExit(f'private path or address would reach the repository: {rel} ({m.group(0)[:24]!r})')
    inst = json.loads((out / 'sources/installed-files.json').read_text())
    for item in inst['profiles'][doc['topology']]['files']:
        copies = [(item['source'], item['sha256'])] if item.get('source') else []
        copies += [(f['source'], f['sha256']) for f in item.get('files', {}).values() if f.get('source')]
        for source, want in copies:
            if sha256((out / source).read_bytes()) != want:
                raise SystemExit('staged copy differs from the served sha256: ' + source)
    listed = {i['source'] for i in inst['profiles'][doc['topology']]['files'] if i.get('source')}
    listed |= {f['source'] for i in inst['profiles'][doc['topology']]['files'] for f in i.get('files', {}).values()
               if f.get('source')}
    listed |= {i['patch'] for i in inst['profiles'][doc['topology']]['files'] if i.get('patch')}
    for d in owned:
        for p in (out / d).rglob('*') if (out / d).exists() else []:
            rel = p.relative_to(out).as_posix()
            if p.is_file() and rel not in listed and not rel.startswith('launch/profiles/'):
                raise SystemExit('staged file not in the manifest: ' + rel)


def commit(root, out, owned, shared):
    """Swap the staged outputs into the repository; on any error, restore what was there."""
    backup = out / '.previous'
    done = []
    try:
        for d in owned:
            target, staged, old = root / d, out / d, backup / d
            if target.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, old)
            done.append(('dir', d))
            if staged.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, target)
        for f in shared:
            target, staged, old = root / f, out / f, backup / f
            if target.exists():
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, old)
            done.append(('file', f))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, target)
    except BaseException:
        for kind, rel in reversed(done):
            target, old = root / rel, backup / rel
            if kind == 'dir':
                if target.exists():
                    shutil.rmtree(target)
                if old.exists():
                    os.replace(old, target)
            elif old.exists():
                os.replace(old, target)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('document', type=Path)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    root = a.root.resolve()
    doc = json.loads(a.document.read_text())
    if doc.get('schema') != 'switchless-import/1' or doc.get('topology') not in ('tp4', 'tp2'):
        raise SystemExit('unsupported import document')
    out = Path(tempfile.mkdtemp(prefix='.import-', dir=root))  # same file system: the swap is a rename
    try:
        owned, shared, summary = build(doc, root, out)
        validate(doc, out, owned, shared)
        commit(root, out, owned, shared)
    finally:
        shutil.rmtree(out, ignore_errors=True)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
