# SPDX-License-Identifier: Apache-2.0
"""Check the recipe's metadata, patches and served-file manifests on a CPU. Never fetches, builds or launches.

    python3 -B sources/verify.py [--root DIR] [--release]

Schema 2 checks that every served file of both profiles is accounted for byte for byte: patches parse, touch only
their declared file and carry the recorded pre/postimage Git blob ids; repository copies have the served sha256;
every overlay mount of a profile is a served file and every served file is mounted; series files, lock and
manifests agree; licence and notice files are present. It also checks:
- digests: every image ID is `sha256:<64 hex>`, a public image reference is pinned by digest;
- mounts: every `--mount` option is a bind mount with exactly `type`, `src`, `dst` and an optional trailing
  `readonly`, overlay mounts are read-only, and the profile's mount table equals the options in its argv;
- launch identity: `launch_sha256` (a digest of the argv template, role arguments, mount table, image and every
  served byte of the profile) is recomputed and must equal the value recorded with the lab plan in the profile, in
  its canonical launch fixture and in release/provenance.json, and the fixture's commands must re-render exactly.
  Changing a flag, a mount or a served file without importing a new plan therefore fails;
- artefacts: every native artefact has one identity (name, target, sha256 or per-rank sha256, source tree) across
  release/assets.json, sources/installed-files.json, sources.lock.json and release/provenance.json.
Exit 0 = consistent.

`--release` always exits 2: publication needs a built and qualified image of each profile (docker/README.md) and the
owner's go, none of which a CPU check can establish. The blockers are printed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {'tp4': 4, 'tp2': 2}
UPSTREAMS = {
    'vllm': 'https://github.com/vllm-project/vllm.git',
    'b12x': 'https://github.com/local-inference-lab/b12x.git',
    'sparse-mla': 'https://github.com/Libertai/vllm-sparse-mla-blackwell.git',
    'nccl': 'https://github.com/NVIDIA/nccl.git',
    'rigmark': 'https://github.com/alexellis/rigmark.git',
}
PATCH_KIND = {'vllm': 'file-overlay', 'b12x': 'file-overlay', 'sparse-mla': 'file-overlay',
              'nccl': 'git-series', 'rigmark': 'git-series'}
LOCK_STATUS = {'measured-profiles-local'}
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'   # the Apache License 2.0 text (licenses/Apache-2.0.txt)
NOTICE_NAMES = ('vllm-project/vllm', 'local-inference-lab/b12x', 'Libertai/vllm-sparse-mla-blackwell', 'NVIDIA/nccl',
                'FujitsuPolycom/sparkring', 'alexellis/switchless-nccl', 'vllm-project/humming', 'zai-org')
# NOTICE must also name every model repository of release/assets.json (the served checkpoint and what it mounts)
ALLOWED_PLACEHOLDERS = {'RANK', 'ROLE_ARGS', 'OVERLAY_ROOT', 'SWITCHLESS_API_PORT', 'SWITCHLESS_CONTAINER_NAME',
                        'SWITCHLESS_DRAFTER_REPO_DIR', 'SWITCHLESS_FABRIC_CIDR', 'SWITCHLESS_HF_CACHE_DIR',
                        'SWITCHLESS_IB_HCA', 'SWITCHLESS_JIT_CACHE_DIR', 'SWITCHLESS_MASTER_ADDR',
                        'SWITCHLESS_MASTER_PORT', 'SWITCHLESS_MODEL_DIR', 'SWITCHLESS_PROFILES_DIR',
                        'SWITCHLESS_TOKENIZER_FILE',
                        'SWITCHLESS_RANK_HOST_IP', 'SWITCHLESS_SOCKET_IFNAME'}
PLACEHOLDER = re.compile(r'\$\{([A-Z0-9_:./-]+)\}')
IMAGE_ID = re.compile(r'sha256:[0-9a-f]{64}')
PUBLIC_REFERENCE = re.compile(r'[a-z0-9][a-z0-9._/:-]*@sha256:[0-9a-f]{64}')
MOUNT_KEYS = ('type', 'src', 'dst')
RELEASE_BLOCKERS = [
    'the base image that docker/Dockerfile rebuilds has not been built on a DGX Spark yet (docker/README.md)',
    'the self-built profile images, with their rebuilt native artefacts, have not been qualified on hardware',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hexhash(value, length):
    return type(value) is str and value != '0' * length and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None


def image_id(value, label):
    require(type(value) is str and IMAGE_ID.fullmatch(value) is not None and value != 'sha256:' + '0' * 64,
            label + ': image ID must be sha256:<64 lowercase hex>')
    return value


def public_reference(value, label):
    require(value is None or (type(value) is str and PUBLIC_REFERENCE.fullmatch(value) is not None),
            label + ': a public image reference must be pinned by digest (NAME@sha256:<64 hex>)')


def served_view(served):
    """What a profile serves, without descriptive fields: every target with its kind and bytes by sha256."""
    view = []
    for item in sorted(served['files'], key=lambda i: i['target']):
        files = {name: {k: f[k] for k in ('sha256', 'per_rank_sha256') if f.get(k)}
                 for name, f in sorted(item.get('files', {}).items())}
        view.append({'target': item['target'], 'kind': item['kind'], 'sha256': item.get('sha256'), 'files': files,
                     'symlinks': item.get('symlinks', {}), 'source_dir': item.get('source_dir')})
    return {'image': served['image'], 'files': view}


def launch_digest(profile, served):
    """sha256 over everything that determines what a profile launches: argv template, role arguments, mount table,
    image and every served byte (sources/installed-files.json, by sha256). Descriptive fields are left out."""
    body = {'topology': profile['topology'], 'ranks': profile['ranks'], 'image': profile['image']['id'],
            'argv_template': profile['argv_template'], 'role_args': profile['role_args'], 'mounts': profile['mounts'],
            'site_variables': profile['site_variables'], 'served': served_view(served)}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_mount(token, label):
    """A docker --mount option as the profiles write it: type=bind,src=...,dst=...[,readonly]."""
    require(type(token) is str and token, label + ': empty mount option')
    parts = token.split(',')
    readonly = parts[-1] == 'readonly'
    kv = {}
    for part in parts[:-1] if readonly else parts:
        key, sep, value = part.partition('=')
        require(sep == '=' and key in MOUNT_KEYS and key not in kv and value, label + ': mount option must be '
                'type=bind,src=...,dst=... with an optional trailing readonly, not ' + repr(token))
        kv[key] = value
    require(set(kv) == set(MOUNT_KEYS) and kv['type'] == 'bind', label + ': bind mount with src and dst required: '
            + repr(token))
    require(kv['dst'].startswith('/') and '..' not in kv['dst'].split('/'), label + ': mount target must be absolute')
    return kv['src'], kv['dst'], readonly


def load_renderer():
    spec = importlib.util.spec_from_file_location('switchless_render', ROOT / 'launch/render.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def relative_name(name):
    require(type(name) is str and bool(name), 'nonempty relative path required')
    require('\\' not in name and not any(ord(c) < 32 or ord(c) == 127 for c in name),
            'path contains a separator alias or control character: ' + repr(name))
    require(not PurePosixPath(name).is_absolute() and all(p not in ('', '.', '..') for p in name.split('/')),
            'normalized relative path required: ' + name)
    return name


def inside(root, name, directory=False):
    """Resolve a repository path; reject symlinks in every component."""
    relative_name(name)
    path = Path(root)
    parts = name.split('/')
    for index, part in enumerate(parts):
        path = path / part
        mode = path.lstat().st_mode
        require(not stat.S_ISLNK(mode), 'symlink is not an input: ' + name)
        last = index == len(parts) - 1
        require((stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)) if last else stat.S_ISDIR(mode),
                'regular file/directory required: ' + name)
    return path


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('non-JSON numeric constant: ' + value)


def read_json(root, name):
    return json.loads(inside(root, name).read_text(encoding='utf-8'), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def fields(value, expected, label, optional=()):
    require(type(value) is dict, label + ': object required')
    keys = set(value)
    require(set(expected) <= keys <= set(expected) | set(optional), label + ': unexpected or missing fields')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def numstat(root, data):
    r = subprocess.run(['git', 'apply', '--numstat'], input=data, cwd=root, capture_output=True, check=False)
    require(r.returncode == 0 and r.stdout.strip(), 'invalid or empty patch')
    return [line.split('\t')[2] for line in r.stdout.decode().splitlines()]


def series(root, component):
    lines = inside(root, f'patches/{component}/series').read_text(encoding='utf-8').splitlines()
    out = [line for line in lines if line and not line.startswith('#')]
    for line in out:
        require(line == line.strip(), component + ': whitespace in series path')
        relative_name(line)
    return out


def check_overlay_patch(root, component, entry):
    fields(entry, ('path', 'sha256', 'target', 'profile', 'preimage_sha256', 'preimage_blob', 'preimage_upstream',
                   'postimage_sha256', 'postimage_blob'), component + ' patch')
    require(entry['profile'] in PROFILES, component + ': unknown profile')
    path = relative_name(entry['path'])
    require(path.startswith(entry['profile'] + '/') and path.endswith('.patch') and path.count('/') == 1,
            component + ': patch must live in its profile directory: ' + path)
    data = inside(root, f'patches/{component}/{path}').read_bytes()
    require(hexhash(entry['sha256'], 64) and sha256(data) == entry['sha256'], component + ': patch hash differs: ' + path)
    for key in ('preimage_sha256', 'postimage_sha256'):
        require(hexhash(entry[key], 64), component + ': bad ' + key)
    for key in ('preimage_blob', 'postimage_blob'):
        require(hexhash(entry[key], 40), component + ': bad ' + key)
    require(type(entry['preimage_upstream']) is str and entry['preimage_upstream'].strip(), component + ': upstream relation')
    relative_name(entry['target'])
    touched = numstat(root, data)
    require(touched == [entry['target']], component + ': patch touches other files: ' + path)
    text = data.decode('utf-8', errors='replace')
    index = re.findall(r'^index ([0-9a-f]{40})\.\.([0-9a-f]{40})', text, re.M)
    require(index == [(entry['preimage_blob'], entry['postimage_blob'])], component + ': index blob ids differ: ' + path)
    require(f"Preimage:    sha256 {entry['preimage_sha256']}" in text
            and f"Postimage:   sha256 {entry['postimage_sha256']}" in text, component + ': header hashes differ: ' + path)
    return path


def check_lock(root):
    lock = read_json(root, 'sources.lock.json')
    fields(lock, ('schema', 'status', 'base_image', 'components'), 'source lock')
    require(type(lock['schema']) is int and lock['schema'] == 2, 'source lock: schema must be integer 2')
    require(lock['status'] in LOCK_STATUS, 'source lock: unsupported status')
    image = lock['base_image']
    fields(image, ('id', 'tag', 'public_reference', 'contents'), 'base image')
    image_id(image['id'], 'base image')
    public_reference(image['public_reference'], 'base image')
    fields(lock['components'], tuple(UPSTREAMS), 'components')
    overlay = {}
    for name in UPSTREAMS:
        item = lock['components'][name]
        optional = ('result_tree',) if PATCH_KIND[name] == 'git-series' else ('build_patches',)
        fields(item, ('upstream', 'base_commit', 'base_tree', 'git_object_format', 'patch_kind', 'patches'), name,
               optional=optional + (('result_tree',) if PATCH_KIND[name] == 'git-series' else ()))
        require(item['upstream'] == UPSTREAMS[name], name + ': canonical public upstream required')
        require(item['git_object_format'] == 'sha1' and hexhash(item['base_commit'], 40) and hexhash(item['base_tree'], 40),
                name + ': invalid recorded base identity')
        require(item['patch_kind'] == PATCH_KIND[name], name + ': unexpected patch kind')
        require(type(item['patches']) is list, name + ': patches must be a list')
        listed = []
        if item['patch_kind'] == 'file-overlay':
            seen = set()
            for entry in item['patches']:
                listed.append(check_overlay_patch(root, name, entry))
                key = (entry['profile'], entry['target'])
                require(key not in seen, name + ': duplicate overlay target')
                seen.add(key)
                overlay[(entry['profile'], entry['target'])] = entry
            for entry in item.get('build_patches', []):
                fields(entry, ('path', 'sha256', 'profile', 'artefact', 'artefact_sha256', 'result_tree'), name + ' build patch')
                p = relative_name(entry['path'])
                require(p.startswith('build/'), name + ': build patches live in build/')
                data = inside(root, f'patches/{name}/{p}').read_bytes()
                require(sha256(data) == entry['sha256'], name + ': build patch hash differs')
                numstat(root, data)
                require(hexhash(entry['artefact_sha256'], 64) and hexhash(entry['result_tree'], 40), name + ': build identity')
                listed.append(p)
        else:
            trees = []
            for entry in item['patches']:
                fields(entry, ('path', 'sha256', 'result_tree'), name + ' patch')
                p = relative_name(entry['path'])
                data = inside(root, f'patches/{name}/{p}').read_bytes()
                require(sha256(data) == entry['sha256'], name + ': patch hash differs: ' + p)
                numstat(root, data)
                require(hexhash(entry['result_tree'], 40), name + ': invalid intermediate tree')
                trees.append(entry['result_tree'])
                listed.append(p)
            final = item.get('result_tree')
            require(final is None if not trees else final == trees[-1], name + ': final tree inconsistent with the series')
        in_series = [p for p in listed if not p.startswith('build/')]
        require(series(root, name) == in_series, name + ': series file disagrees with the lock')
        allowed = set(listed) | {'series', 'README.md', 'provenance.json'}
        folder = root / 'patches' / name
        for path in folder.rglob('*'):
            rel = path.relative_to(folder).as_posix()
            require(not path.is_symlink(), name + ': symlink in patch queue')
            if not path.is_dir():
                require(rel in allowed, name + ': unlisted patch queue file: ' + rel)
    return lock, overlay


def check_manifest(root, lock, overlay):
    inst = read_json(root, 'sources/installed-files.json')
    fields(inst, ('schema', 'install_root', 'profiles'), 'installed files')
    require(inst['schema'] == 2 and inst['install_root'] == '/usr/local/lib/python3.12/dist-packages/',
            'installed files: schema/install root')
    fields(inst['profiles'], tuple(PROFILES), 'installed profiles')
    served = {}
    for topo, prof in inst['profiles'].items():
        fields(prof, ('image', 'files', 'artefacts'), topo + ' manifest')
        require(image_id(prof['image'], topo + ' manifest') == lock['base_image']['id'],
                topo + ': manifest image differs from the lock')
        targets = {}
        for item in prof['files']:
            require(type(item) is dict and item.get('target', '').startswith('/'), topo + ': absolute target required')
            t = item['target']
            require(t not in targets, topo + ': duplicate target ' + t)
            kind = item.get('kind')
            if kind in ('patch', 'src', 'template'):
                require(hexhash(item.get('sha256'), 64), topo + ': served sha256 required for ' + t)
                require(type(item.get('licence')) is str and item['licence'].strip(), topo + ': licence required for ' + t)
            if kind == 'patch':
                rel = t[len(inst['install_root']):]
                entry = overlay.get((topo, rel))
                require(entry is not None and entry['postimage_sha256'] == item['sha256']
                        and item['patch'] == f"patches/{item['patch'].split('/')[1]}/{entry['path']}"
                        and item['preimage_sha256'] == entry['preimage_sha256'], topo + ': patch entry disagrees: ' + t)
            elif kind in ('src', 'template'):
                data = inside(root, item['source']).read_bytes()
                require(sha256(data) == item['sha256'], topo + ': repository copy differs from the served file: ' + t)
                if 'lab_served_sha256' in item:   # the SPDX line changed with the repository licence, recorded per file
                    require(hexhash(item['lab_served_sha256'], 64) and item['lab_served_sha256'] != item['sha256']
                            and type(item.get('recipe_change')) is str and b'SPDX-License-Identifier: Apache-2.0' in data[:600],
                            topo + ': a relicensed file needs its lab sha256, the change and the Apache-2.0 SPDX line: ' + t)
                if kind == 'src':
                    require(item['source'] == f'src/{topo}/' + t[len(inst['install_root']):], topo + ': src path mismatch ' + t)
            elif kind in ('src-dir', 'config-dir', 'binary-dir', 'third-party-dir'):
                require(type(item.get('files')) is dict and item['files'], topo + ': empty directory entry ' + t)
                for name, f in item['files'].items():
                    relative_name(name)
                    require(hexhash(f.get('sha256'), 64), topo + ': sha256 required for ' + t + '/' + name)
                    if f.get('per_rank_sha256') is not None:
                        per_rank = f['per_rank_sha256']
                        require(type(per_rank) is dict and sorted(per_rank) == [str(r) for r in range(PROFILES[topo])]
                                and all(hexhash(v, 64) for v in per_rank.values()) and f['sha256'] in per_rank.values(),
                                topo + ': per-rank sha256 of ' + t + '/' + name)
                    if f.get('source'):
                        require(sha256(inside(root, f['source']).read_bytes()) == f['sha256'], topo + ': copy differs ' + f['source'])
                if kind == 'third-party-dir':
                    source_dir = item.get('source_dir')
                    require(type(source_dir) is str and relative_name(source_dir) == source_dir and '/' not in source_dir,
                            topo + ': a third-party directory must name its source directory: ' + t)
                    dist = item.get('distribution')
                    if source_dir.endswith('.dist-info'):
                        require(type(dist) is dict and set(dist) == {'name', 'version'} and source_dir
                                == dist['name'].replace('-', '_') + '-' + dist['version'] + '.dist-info'
                                and item.get('artefact') == dist['name'] + '-' + dist['version'],
                                topo + ': dist-info source directory, distribution and artefact disagree: ' + t)
            elif kind == 'binary':
                require(hexhash(item.get('sha256'), 64) and item.get('artefact'), topo + ': binary identity ' + t)
            else:
                raise ValueError(topo + ': unknown kind for ' + t)
            targets[t] = item
        served[topo] = targets
        for (p, rel), entry in overlay.items():
            if p == topo:
                require(inst['install_root'] + rel in targets, topo + ': lock patch not in the manifest: ' + rel)
    return inst, served


def check_profiles(root, lock, inst, served, provenance):
    renderer = load_renderer()
    for topo, ranks in PROFILES.items():
        base = f'launch/profiles/{topo}/'
        prof = read_json(root, base + 'profile.json')
        fields(prof, ('schema', 'topology', 'ranks', 'status', 'deterministic', 'nondeterministic_note', 'source_plan',
                      'image', 'argv_template', 'role_args', 'mounts', 'site_variables', 'overlay'), topo + ' profile')
        require(prof['schema'] == 2 and prof['topology'] == topo and prof['ranks'] == ranks, topo + ': profile identity')
        fields(prof['image'], ('id', 'tag', 'public_reference'), topo + ' profile image')
        require(image_id(prof['image']['id'], topo + ' profile') == lock['base_image']['id'],
                topo + ': profile image differs from the lock')
        public_reference(prof['image']['public_reference'], topo + ' profile')
        plan = prof['source_plan']
        fields(plan, ('sha256', 'label', 'render_check', 'launch_sha256'), topo + ' source plan',
               optional=('derived_from', 'qualification'))
        require(hexhash(plan['sha256'], 64) and hexhash(plan['launch_sha256'], 64), topo + ': plan identity')
        if 'qualification' in plan:  # evidence of a derived plan must be that plan's and name what it did not establish
            q = plan['qualification']
            require(type(q) is dict and q.get('plan_sha256') == plan['sha256'] and 'derived_from' in plan
                    and type(q.get('evidence')) is list and type(q.get('not_established')) is list,
                    topo + ': the qualification must be the derived plan\'s own, with evidence and not_established lists')
        require(prof['deterministic'] is None or type(prof['deterministic']) is bool, topo + ': deterministic flag')
        if prof['deterministic'] is False:
            require(type(prof['nondeterministic_note']) is str and 'NON-DETERMINISTIC' in prof['nondeterministic_note'],
                    topo + ': a non-deterministic profile must say so')
        argv = prof['argv_template']
        require(type(argv) is list and argv[:2] == ['docker', 'run'] and all(type(t) is str for t in argv), topo + ': argv')
        require(argv.count(prof['image']['id']) == 1 and argv.count('${ROLE_ARGS}') == 1, topo + ': image/role token')
        used = {m.group(1) for tok in argv + prof['role_args']['head'] for m in PLACEHOLDER.finditer(tok)}
        require(used <= ALLOWED_PLACEHOLDERS, topo + ': undeclared placeholder ' + str(sorted(used - ALLOWED_PLACEHOLDERS)))
        require(set(prof['site_variables']) == used - {'RANK', 'ROLE_ARGS', 'OVERLAY_ROOT'}, topo + ': site variable list')
        mounted, table = set(), {}
        for i, tok in enumerate(argv):
            if i and argv[i - 1] == '--mount':
                src, dst, readonly = parse_mount(tok, topo)
                require(dst not in table, topo + ': mounted twice ' + dst)
                table[dst] = {'source': src, 'readonly': readonly}
                if src.startswith('${OVERLAY_ROOT}'):
                    require(src == '${OVERLAY_ROOT}' + dst, topo + ': overlay source must mirror its target ' + dst)
                    require(readonly, topo + ': overlay mounts must be read-only: ' + dst)
                    require(dst in served[topo], topo + ': mounted overlay file is not in the manifest: ' + dst)
                    mounted.add(dst)
                else:
                    require(PLACEHOLDER.fullmatch(src) is not None, topo + ': site mount must be one variable ' + dst)
        require(mounted == set(served[topo]), topo + ': served files not mounted: ' + str(sorted(set(served[topo]) - mounted)))
        require(prof['mounts'] == table, topo + ': the mount table differs from the --mount options of the argv '
                + str(sorted(k for k in set(table) | set(prof['mounts']) if table.get(k) != prof['mounts'].get(k))[:3]))
        override = inside(root, base + 'override.json').read_bytes()
        cfg = served[topo].get('/opt/glm53-speedup/runtime')
        require(cfg is not None and cfg['files']['override.json']['sha256'] == sha256(override), topo + ': override differs')

        # launch identity: recomputed, recorded with the plan in three places, and the canonical commands re-render
        digest = launch_digest(prof, inst['profiles'][topo])
        fixture = read_json(root, base + 'canonical-launch.json')
        fields(fixture, ('schema', 'topology', 'source_plan_sha256', 'launch_sha256', 'note', 'site', 'argv'),
               topo + ' canonical launch')
        record = provenance['profiles'].get(topo) or {}
        require(plan['launch_sha256'] == digest and fixture['launch_sha256'] == digest
                and record.get('launch_sha256') == digest,
                topo + ': the launch (flags, mounts or served files) differs from the one recorded with lab plan '
                + plan['sha256'][:12] + '; import a new plan (docs/updating.md)')
        require(fixture['topology'] == topo and fixture['source_plan_sha256'] == plan['sha256']
                and record.get('lab_plan_sha256') == plan['sha256'], topo + ': lab plan identity differs between the '
                'profile, its canonical launch and release/provenance.json')
        require(type(fixture['argv']) is dict and sorted(fixture['argv']) == [str(r) for r in range(ranks)],
                topo + ': canonical launch ranks')
        for r in range(ranks):
            require(renderer.render(prof, fixture['site'], r) == fixture['argv'][str(r)],
                    topo + f': rank {r} no longer renders to the canonical launch')


def check_licences(root, inst):
    require(sha256(inside(root, 'LICENSE').read_bytes()) == LICENSE_SHA256, 'root Apache-2.0 license text differs')
    notice = inside(root, 'NOTICE').read_text(encoding='utf-8')
    models = [m['name'] for m in read_json(root, 'release/assets.json')['models']]
    for name in NOTICE_NAMES + tuple(models):
        require(name in notice, 'NOTICE lacks ' + name)
    comp = read_json(root, 'licenses/components.json')
    fields(comp, ('schema', 'original_code', 'components'), 'licence components')
    require(comp['schema'] == 2 and comp['original_code'] == 'Apache-2.0', 'licence scope')
    for name, entry in comp['components'].items():
        fields(entry, ('upstream', 'licence', 'texts', 'used_by'), 'licence ' + name, optional=('notes',))
        for text in entry['texts']:
            inside(root, text)
    for topo, prof in inst['profiles'].items():
        for item in prof['files']:
            for key in item.get('attribution', []):
                require(key in comp['components'], topo + ': attribution key without licence entry: ' + key)


def check_artefacts(root, lock, inst, provenance):
    """One identity per native artefact across release/assets.json, the served-file manifest, the lock and
    release/provenance.json."""
    assets = read_json(root, 'release/assets.json')
    fields(assets, ('schema', 'image', 'artefacts', 'models', 'note'), 'release assets')
    fields(assets['image'], ('id', 'tag', 'public_reference', 'status'), 'release assets image')
    require(image_id(assets['image']['id'], 'release assets') == lock['base_image']['id'], 'release assets: image differs')
    public_reference(assets['image']['public_reference'], 'release assets')
    names = {}
    for a in assets['artefacts']:
        fields(a, ('name', 'profiles', 'source', 'build', 'published'), 'asset ' + str(a.get('name')),
               optional=('target', 'targets', 'sha256', 'bytes', 'per_rank_sha256', 'built_from_tree', 'note'))
        require(a['name'] not in names, 'release assets: duplicate artefact ' + a['name'])
        require(('target' in a) != ('targets' in a), a['name'] + ': one of target / targets')
        if 'sha256' in a:
            require(hexhash(a['sha256'], 64), a['name'] + ': invalid sha256')
        if 'per_rank_sha256' in a:
            require(type(a['per_rank_sha256']) is dict and all(hexhash(v, 64) for v in a['per_rank_sha256'].values()),
                    a['name'] + ': invalid per-rank sha256')
        names[a['name']] = a
    def same_bytes(a, sha, per_rank):
        """The asset's recorded bytes: one sha256, or one per rank (then `sha` is one of them)."""
        if a.get('per_rank_sha256') is not None or per_rank:
            return a.get('per_rank_sha256') == per_rank and sha in per_rank.values() and 'sha256' not in a
        return a.get('sha256') == sha

    used = set()
    for topo, prof in inst['profiles'].items():
        for art in prof['artefacts']:
            a = names.get(art['name'])
            require(a is not None, topo + ': artefact not described in release/assets.json: ' + art['name'])
            require(topo in a['profiles'], topo + ': release/assets.json does not list the profile for ' + art['name'])
            used.add(art['name'])
            if art.get('sha256'):
                require(same_bytes(a, art['sha256'], art.get('per_rank_sha256')),
                        art['name'] + ': sha256 differs between the manifests')
            if 'target' in a and not a.get('targets'):
                require(a['target'] == art['target'] or a['target'].startswith(art['target'] + '/'),
                        art['name'] + ': target differs between the manifests')
        for item in prof['files']:
            if item['kind'] == 'binary':
                a = names.get(item['artefact'])
                require(a is not None and a.get('sha256') == item['sha256'] and a.get('target') == item['target']
                        and (item.get('bytes') is None or a.get('bytes') == item['bytes']),
                        topo + ': binary identity differs from release/assets.json: ' + item['target'])
                require({'name': item['artefact'], 'target': item['target'], 'sha256': item['sha256']} in prof['artefacts'],
                        topo + ': binary not in the profile artefact list: ' + item['target'])
            for name, f in item.get('files', {}).items():
                if f.get('artefact'):
                    a = names.get(f['artefact'])
                    require(a is not None and same_bytes(a, f['sha256'], f.get('per_rank_sha256'))
                            and a.get('target') == item['target'] + '/' + name,
                            topo + ': artefact identity differs from release/assets.json: ' + item['target'] + '/' + name)
            if item['kind'] == 'binary-dir':
                a = names.get(item.get('artefact'))
                require(a is not None and len(item['files']) == 1, topo + ': binary directory artefact ' + item['target'])
                (name, f), = item['files'].items()
                require(a.get('sha256') == f['sha256'] and a.get('target') == item['target'] + '/' + name,
                        topo + ': artefact identity differs from release/assets.json: ' + item['target'] + '/' + name)
            if item['kind'] == 'third-party-dir':
                a = names.get(item.get('artefact'))
                require(a is not None and (a.get('targets') or {}).get(item['target']) == item.get('source_dir'),
                        topo + ': third-party directory and its source directory differ from release/assets.json: '
                        + item['target'])
    require(used == set(names), 'release/assets.json lists artefacts no profile serves: ' + str(sorted(set(names) - used)))
    trees = provenance['source_trees']
    for entry in lock['components']['sparse-mla'].get('build_patches', []):
        a = names.get(entry['artefact'])
        require(a is not None and a.get('sha256') == entry['artefact_sha256'],
                entry['artefact'] + ': build patch artefact sha256 differs from release/assets.json')
        require(trees.get(entry['artefact']) == entry['result_tree'], entry['artefact'] + ': source tree differs '
                'between sources.lock.json and release/provenance.json')
    nccl = lock['components']['nccl']['result_tree']
    require(trees.get('nccl') == nccl, 'nccl: source tree differs between sources.lock.json and release/provenance.json')
    require(nccl in names['nccl-switchless-dual-pf']['source'], 'nccl: release/assets.json must name the series tree')
    for a in names.values():  # a measured artefact built from an earlier tree names that tree
        if 'built_from_tree' in a:
            require(hexhash(a['built_from_tree'], 40) and a['built_from_tree'] in a['source'],
                    a['name'] + ': release/assets.json must name the source tree its bytes were built from')


def check_provenance(root):
    provenance = read_json(root, 'release/provenance.json')
    fields(provenance, ('schema', 'recipe', 'prepared', 'profiles', 'source_trees', 'checks'), 'release provenance',
           optional=('selfbuilt',))   # the record of a build of docker/Dockerfile and its qualification
    fields(provenance['profiles'], tuple(PROFILES), 'release provenance profiles')
    for topo, entry in provenance['profiles'].items():
        require(hexhash(entry.get('lab_plan_sha256'), 64) and hexhash(entry.get('launch_sha256'), 64),
                topo + ': release/provenance.json needs the lab plan and launch sha256')
    for name, tree in provenance['source_trees'].items():
        require(hexhash(tree, 40), 'release provenance: invalid source tree for ' + name)
    if 'selfbuilt' in provenance:
        sb = provenance['selfbuilt']
        require(type(sb) is dict and image_id(sb.get('image_tp4'), 'self-built TP4 image') and sb.get('date'),
                'release provenance: the self-built record needs its date and TP4 image ID')
    return provenance


def check(root):
    root = Path(root).resolve()
    lock, overlay = check_lock(root)
    inst, served = check_manifest(root, lock, overlay)
    provenance = check_provenance(root)
    check_profiles(root, lock, inst, served, provenance)
    check_licences(root, inst)
    check_artefacts(root, lock, inst, provenance)
    return RELEASE_BLOCKERS.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--release', action='store_true', help='always fails: see the blockers')
    args = parser.parse_args()
    try:
        blockers = check(args.root)
    except (ValueError, OSError, KeyError, TypeError, RecursionError) as exc:
        print(json.dumps({'status': 'INVALID', 'error': str(exc)}))
        return 2
    print(json.dumps({'status': 'RELEASE_BLOCKED' if args.release else 'CONSISTENT',
                      'release_blockers': blockers, 'gpu_or_serving_checked': False}, indent=2))
    return 2 if args.release else 0


if __name__ == '__main__':
    raise SystemExit(main())
