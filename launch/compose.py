# SPDX-License-Identifier: Apache-2.0
"""Render one Docker Compose project per rank for the self-built profile image. Writes files only; runs nothing.

    python3 -B launch/compose.py --profile tp4 --site site.env --out deploy/tp4
    python3 -B launch/compose.py --profile tp4 --site site.env --plan      # the per-rank plan launch/up.sh reads

For every rank, OUT/rank<N>/ gets
- compose.yaml: the service `glm53` with the docker-run semantics of `baked_argv()` (image, bind mounts, environment,
  network, IPC, GPUs, devices, capabilities, security options, ulimits, shared memory, labels, container name and the
  server command, in the argv's order), and a `build:` section for docker/Dockerfile's `profile-<profile>` target; the
  build-only service `vllm-source` (compose profile `build`) builds the vLLM source image the Dockerfile starts from.
  The file is JSON, which is valid YAML; Docker Compose reads it as it is.
- .env: the rank's site values that compose.yaml refers to as ${SWITCHLESS_...}; Compose reads it from the project
  directory. It holds site data, like site.env: keep it out of Git (deploy/ is ignored).
The build context is `../../..`: keep each project directory at deploy/<profile>/rank<N> of a clone of this
repository (launch/up.sh copies it there on every node).

`baked_argv()` is the canonical `docker run` of launch/render.py for the profile image that docker/Dockerfile builds:
the same tokens, without the `${OVERLAY_ROOT}` mounts (that image carries the same files at the same paths) and with
the measured base image ID replaced by the self-built image (SWITCHLESS_BUILD_IMAGE, default
glm53-switchless:<profile>). The measured launches used the base image and the mounts; a self-built image is a new
candidate and needs its own qualification (docker/README.md).

Site values are checked exactly as launch/render.py checks them, and the optional variants it turns on
(SWITCHLESS_WEIGHTLESS_DIR) apply here too: their mounts and variables join the service, their site values the .env.
For launch/up.sh: SWITCHLESS_SSH_TARGETS (one ssh target per rank; default SWITCHLESS_RANK_HOST_IPS; per rank
SWITCHLESS_RANK<N>_SSH), SWITCHLESS_REMOTE_REPO (this repository's path on every node; default this checkout's path),
SWITCHLESS_SHARE_TARGETS (optional: the other ranks as the build host reaches them, for sharing the image directly),
SWITCHLESS_API_HOST (optional: the address clients use for rank 0).
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('switchless_render', ROOT / 'launch/render.py')
render = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(render)

SERVICE = 'glm53'
SOURCE_SERVICE = 'vllm-source'
SOURCE_IMAGE = 'glm53-switchless/vllm-glm-release:4500c80c'
# The vLLM source image: ZJY0516/vllm's own docker/Dockerfile at the glm-release commit the lab built from, with the
# lab's build arguments. BUILD_BASE_IMAGE is the lab's ARM64 fix (its only change to that Dockerfile); the two base
# images are pinned by the digests the lab's build resolved (docs/build-provenance.md).
SOURCE_BUILD = {
    'context': 'https://github.com/ZJY0516/vllm.git#4500c80c080328dfe62435d083f4063e00d987df',
    'dockerfile': 'docker/Dockerfile',
    'target': 'vllm-openai',
    'args': {
        'CUDA_VERSION': '13.0.3',
        'PYTHON_VERSION': '3.12',
        'torch_cuda_arch_list': '12.1a',
        'FLASHINFER_VERSION': '0.6.18',
        'max_jobs': '16',
        'nvcc_threads': '2',
        'BUILD_BASE_IMAGE': 'pytorch/manylinuxaarch64-builder:cuda13.0-78e737ad29420ffc4800e677c51e2a852caf8359'
                            '@sha256:f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9',
        'FINAL_BASE_IMAGE': 'nvidia/cuda:13.0.3-base-ubuntu24.04'
                            '@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1',
        'BUILDKIT_CONTEXT_KEEP_GIT_DIR': '1',
    },
}
CONTEXT = '../../..'  # the project directory belongs at deploy/<profile>/rank<N> of a clone of this repository
VALUE_FLAGS = ('--name', '--network', '--ipc', '--gpus', '--cap-add', '--security-opt', '--device', '--ulimit',
               '--shm-size', '--label', '--mount', '-e')


def load_profile(name):
    return json.loads((ROOT / 'launch/profiles' / name / 'profile.json').read_text())


def build_image(site, profile):
    return site.get('SWITCHLESS_BUILD_IMAGE') or f"glm53-switchless:{profile['topology']}"


def baked_profile(profile, image):
    """The argv template for the baked image: `${OVERLAY_ROOT}` mounts dropped, the base image ID replaced."""
    argv, out, i = profile['argv_template'], [], 0
    while i < len(argv):
        if argv[i] == '--mount' and argv[i + 1].startswith('type=bind,src=${OVERLAY_ROOT}/'):
            i += 2
            continue
        out.append(image if argv[i] == profile['image']['id'] else argv[i])
        i += 1
    return dict(profile, argv_template=out, image=dict(profile['image'], id=image))


def baked_argv(profile, site, rank, image=None):
    """launch/render.py's docker run for the baked profile image (SWITCHLESS_IMAGE does not apply to it)."""
    image = image or build_image(site, profile)
    site = {k: v for k, v in site.items() if k != 'SWITCHLESS_IMAGE'}
    return render.render(baked_profile(render.with_options(profile, site), image), site, rank)


def compose_token(tok):
    """A template token as Compose text: site placeholders become ${SWITCHLESS_...} references to .env, `${RANK}`
    becomes ${SWITCHLESS_RANK}, and every other `$` is escaped as `$$`."""
    parts, last = [], 0
    for m in render.PLACEHOLDER.finditer(tok):
        parts.append(tok[last:m.start()].replace('$', '$$'))
        key = m.group(1)
        if key == 'RANK':
            key = 'SWITCHLESS_RANK'
        if not key.startswith('SWITCHLESS_'):
            raise ValueError('placeholder without a site value: ' + m.group(0))
        parts.append('${' + key + '}')
        last = m.end()
    parts.append(tok[last:].replace('$', '$$'))
    return ''.join(parts)


def service(profile, rank, image, context):
    """The Compose service of one rank: the baked argv template, token by token, as Compose keys."""
    tpl = baked_profile(profile, image)
    argv = []
    for tok in tpl['argv_template']:
        if tok == '${ROLE_ARGS}':
            argv += tpl['role_args']['head' if rank == 0 else 'worker']
        else:
            argv.append(tok)
    if argv[:2] != ['docker', 'run'] or argv.count(image) != 1:
        raise ValueError('unexpected argv template')
    svc = {'image': '${SWITCHLESS_BUILD_IMAGE}',
           'build': {'context': context, 'dockerfile': 'docker/Dockerfile', 'target': f"profile-{profile['topology']}",
                     'additional_contexts': {SOURCE_SERVICE: f'service:{SOURCE_SERVICE}'}},
           'environment': [], 'volumes': []}
    i, end = 2, argv.index(image)
    while i < end:
        flag = argv[i]
        if flag == '-d':
            i += 1  # detached: launch/up.sh runs `docker compose up -d`
            continue
        if flag not in VALUE_FLAGS:
            raise ValueError('docker run option without a Compose mapping: ' + flag)
        value = compose_token(argv[i + 1])
        i += 2
        if flag == '-e':
            svc['environment'].append(value)
        elif flag == '--mount':
            src, dst, readonly = parse_mount(value)
            svc['volumes'].append({'type': 'bind', 'source': src, 'target': dst, 'read_only': readonly,
                                   'bind': {'create_host_path': False}})
        elif flag == '--name':
            svc['container_name'] = value
        elif flag == '--network':
            svc['network_mode'] = value
        elif flag == '--ipc':
            svc['ipc'] = value
        elif flag == '--gpus':
            if value != 'all':
                raise ValueError('--gpus ' + value)
            svc['deploy'] = {'resources': {'reservations': {'devices': [
                {'driver': 'nvidia', 'count': 'all', 'capabilities': ['gpu']}]}}}
        elif flag == '--cap-add':
            svc.setdefault('cap_add', []).append(value)
        elif flag == '--security-opt':
            svc.setdefault('security_opt', []).append(value)
        elif flag == '--device':
            svc.setdefault('devices', []).append(f'{value}:{value}')
        elif flag == '--ulimit':
            name, _, limits = value.partition('=')
            soft, _, hard = limits.partition(':')
            svc.setdefault('ulimits', {})[name] = {'soft': int(soft), 'hard': int(hard or soft)}
        elif flag == '--shm-size':
            svc['shm_size'] = value
        elif flag == '--label':
            svc.setdefault('labels', []).append(value)
    svc['command'] = [compose_token(t) for t in argv[end + 1:]]
    return svc


def parse_mount(value):
    parts = value.split(',')
    readonly = parts[-1] == 'readonly'
    kv = dict(p.split('=', 1) for p in (parts[:-1] if readonly else parts))
    if kv.get('type') != 'bind' or set(kv) != {'type', 'src', 'dst'}:
        raise ValueError('unexpected --mount option ' + value)
    return kv['src'], kv['dst'], readonly


def env_values(profile, site, rank, image):
    """The .env of one rank: every site value its compose.yaml refers to, resolved for the rank."""
    values = render.rank_values(site, str(rank), profile['ranks'])
    out = {'SWITCHLESS_BUILD_IMAGE': image, 'SWITCHLESS_RANK': str(rank)}
    for key in sorted(set(profile['site_variables'])):
        out[key] = values.get(key, '')
    return out


def env_text(profile_name, rank, values):
    lines = [f'# Rendered by launch/compose.py for rank {rank} of the {profile_name} profile from the site file.',
             '# Site data: keep it out of Git. Docker Compose reads this file from the project directory.']
    for key, value in values.items():
        if "'" in value or '\n' in value:
            raise ValueError(f'{key}: a quote or newline cannot be written to .env: {value!r}')
        lines.append(f"{key}='{value}'")
    return '\n'.join(lines) + '\n'


def project(profile, site, rank, context):
    image = build_image(site, profile)
    profile = render.with_options(profile, site)  # the optional variants the site turns on (launch/render.py)
    render.render(baked_profile(profile, image), {k: v for k, v in site.items() if k != 'SWITCHLESS_IMAGE'}, rank)
    name = f"switchless-{profile['topology']}-rank{rank}"
    doc = {'name': name, 'services': {
        SERVICE: service(profile, rank, image, context),
        SOURCE_SERVICE: {'profiles': ['build'], 'image': SOURCE_IMAGE, 'build': SOURCE_BUILD}}}
    return doc, env_values(profile, site, rank, image)


def compose_text(doc, profile_name, rank):
    head = [f'# Docker Compose project of rank {rank}, profile {profile_name}: rendered by launch/compose.py; JSON is '
            'valid YAML.',
            '# Build (one node): docker compose --profile build build. Start: docker compose up -d --no-build '
            f'--pull never {SERVICE}',
            '# (workers before rank 0; launch/up.sh does both). docker/README.md and docs/operations.md explain.']
    return '\n'.join(head) + '\n' + json.dumps(doc, indent=2) + '\n'


def resolve(text, env):
    """Compose interpolation of ${VAR} and $$ (what `docker compose config` shows), for the checks."""
    def sub(m):
        if m.group(0) == '$$':
            return '$'
        if m.group(1) not in env:
            raise KeyError(m.group(1))
        return env[m.group(1)]
    return re.sub(r'\$\$|\$\{([A-Z0-9_]+)\}', sub, text)


def split_list(site, key, ranks):
    items = [x.strip() for x in site.get(key, '').split(',') if x.strip()]
    if items and len(items) != ranks:
        raise ValueError(f'{key} lists {len(items)} entries; this profile has {ranks} ranks')
    return items


def plan(profile, site, out_dir):
    """Tab-separated lines for launch/up.sh and launch/down.sh."""
    ranks = profile['ranks']
    targets = split_list(site, 'SWITCHLESS_SSH_TARGETS', ranks) or split_list(site, 'SWITCHLESS_RANK_HOST_IPS', ranks)
    share = split_list(site, 'SWITCHLESS_SHARE_TARGETS', ranks)
    remote = site.get('SWITCHLESS_REMOTE_REPO') or str(ROOT)
    image = build_image(site, profile)
    rows = [('profile', profile['topology']), ('image', image), ('remote_repo', remote),
            ('api_port', site.get('SWITCHLESS_API_PORT', '')),
            ('model_name', profile['argv_template'][profile['argv_template'].index('--served-model-name') + 1])]
    rank0_target = None
    for rank in range(ranks):
        baked_argv(profile, site, rank, image)  # the same checks as launch/render.py
        target = site.get(f'SWITCHLESS_RANK{rank}_SSH') or (targets[rank] if targets else '')
        if not target:
            raise ValueError(f'no ssh target for rank {rank}: set SWITCHLESS_SSH_TARGETS (one per rank)')
        rank0_target = rank0_target or target
        v = render.rank_values(site, str(rank), ranks)
        rel = f'deploy/{profile["topology"]}/rank{rank}'
        rows.append(('rank', str(rank), target, str(Path(out_dir) / f'rank{rank}'), f'{remote}/{rel}',
                     v['SWITCHLESS_CONTAINER_NAME'], v['SWITCHLESS_JIT_CACHE_DIR'], v['SWITCHLESS_PROFILES_DIR'],
                     v['SWITCHLESS_HF_CACHE_DIR'], v['SWITCHLESS_MODEL_DIR'], v['SWITCHLESS_DRAFTER_REPO_DIR'],
                     share[rank] if share else '-'))
    rows.append(('api_host', site.get('SWITCHLESS_API_HOST') or rank0_target.rpartition('@')[2]))
    for row in rows:
        if any('\t' in x or '\n' in x for x in row):
            raise ValueError('tab or newline in ' + row[0])
    return '\n'.join('\t'.join(row) for row in rows) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True, choices=('tp4', 'tp2'))
    ap.add_argument('--site', required=True, type=Path)
    ap.add_argument('--out', type=Path, help='directory for rank<N>/compose.yaml and .env (default deploy/<profile>)')
    ap.add_argument('--plan', action='store_true', help='print the plan for launch/up.sh instead of writing files')
    a = ap.parse_args()
    profile = load_profile(a.profile)
    out = (a.out or ROOT / 'deploy' / a.profile).resolve()
    try:
        site = render.read_site(a.site)
        if a.plan:
            sys.stdout.write(plan(profile, site, out))
            return 0
        written = []
        for rank in range(profile['ranks']):
            rank_dir = out / f'rank{rank}'
            doc, env = project(profile, site, rank, CONTEXT)
            rank_dir.mkdir(parents=True, exist_ok=True)
            (rank_dir / 'compose.yaml').write_text(compose_text(doc, a.profile, rank))
            (rank_dir / '.env').write_text(env_text(a.profile, rank, env))
            written.append(str(rank_dir))
    except ValueError as exc:
        print('compose: ' + str(exc), file=sys.stderr)
        return 2
    print('\n'.join(f'wrote {d}/compose.yaml and .env' for d in written))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
