# SPDX-License-Identifier: Apache-2.0
"""Render the per-rank `docker run` argv of a profile from a site file. Prints only; never runs anything.

    python3 -B launch/render.py --profile tp4 --site site.env [--format json|sh] [--rank N]

The site file holds KEY=VALUE lines (see site.env.example); a value may be quoted with matching single or double
quotes, and an unquoted value may be followed by a ` # comment`. Per-rank values come from
SWITCHLESS_RANK_HOST_IPS (comma list, one per rank) and SWITCHLESS_JIT_CACHE_ROOT / SWITCHLESS_PROFILES_ROOT
(a `rank<N>` directory each), or from explicit SWITCHLESS_RANK<N>_HOST_IP / _JIT_CACHE_DIR / _PROFILES_DIR
overrides. OVERLAY_ROOT is SWITCHLESS_OVERLAY_ROOT: the directory `sources/apply.py` built for this profile.

Refused: a value that would change the meaning of a `--mount` option (a comma in a mounted path), control
characters, and a SWITCHLESS_IMAGE that is not pinned by digest (`sha256:<64 hex>` or `NAME@sha256:<64 hex>`).

Optional, off by default: SWITCHLESS_WEIGHTLESS_DIR (TP4 only) turns on the weightless GLP-44 steering that
launch/options/weightless/prepare.py prepares: DIR/model.py replaces the served model.py mount, and DIR/control.gguf
is mounted at /opt/weightless/control.gguf with the four WEIGHTLESS_* variables, just before the image. Without it,
the output is the profile's.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r'\$\{([A-Z0-9_:./-]+)\}')
PINNED_IMAGE = re.compile(r'(?:[^\s@]+@)?sha256:[0-9a-f]{64}')

# The optional weightless GLP-44 steering (launch/options/weightless/prepare.py, docs/operations.md).
WEIGHTLESS_SITE = 'SWITCHLESS_WEIGHTLESS_DIR'
WEIGHTLESS_MODEL_DST = '/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py'
WEIGHTLESS_VECTOR_DST = '/opt/weightless/control.gguf'
WEIGHTLESS_ENV = ('WEIGHTLESS_STEER_PATH=' + WEIGHTLESS_VECTOR_DST, 'WEIGHTLESS_STEER_ALPHA=2.0',
                  'WEIGHTLESS_STEER_HOOK=post_layer',
                  'WEIGHTLESS_VECTOR_SHA256=0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c')


def with_options(profile, site):
    """The profile with the optional variants the site file turns on; without them, the profile itself.

    SWITCHLESS_WEIGHTLESS_DIR (TP4 only): the served model.py mount takes its file from that directory (in place),
    and the vector mount and the four WEIGHTLESS_* variables come just before the image."""
    if not site.get(WEIGHTLESS_SITE):
        return profile
    if profile['topology'] != 'tp4':
        raise ValueError(f"{WEIGHTLESS_SITE}: the weightless option exists for the tp4 profile only")
    argv = list(profile['argv_template'])
    if any(tok.startswith('WEIGHTLESS_') for tok in argv):
        return profile
    hits = [i for i in range(len(argv) - 1) if argv[i] == '--mount'
            and f',dst={WEIGHTLESS_MODEL_DST},' in argv[i + 1] + ',']
    if len(hits) != 1:
        raise ValueError(f'{WEIGHTLESS_SITE}: the profile has {len(hits)} model.py mounts, expected 1')
    argv[hits[0] + 1] = (f'type=bind,src=${{{WEIGHTLESS_SITE}}}/model.py,dst={WEIGHTLESS_MODEL_DST},readonly')
    extra = ['--mount', f'type=bind,src=${{{WEIGHTLESS_SITE}}}/control.gguf,dst={WEIGHTLESS_VECTOR_DST},readonly']
    for env in WEIGHTLESS_ENV:
        extra += ['-e', env]
    image = argv.index(profile['image']['id'])
    argv[image:image] = extra
    return dict(profile, argv_template=argv,
                site_variables=sorted(set(profile.get('site_variables', [])) | {WEIGHTLESS_SITE}))


def site_value(raw, where):
    """One KEY=VALUE value: matching quotes are stripped; an unquoted value ends at ` #` (a trailing comment)."""
    raw = raw.strip()
    if raw[:1] in ('"', "'"):
        end = raw.find(raw[0], 1)
        rest = raw[end + 1:].strip() if end > 0 else None
        if end < 0 or (rest and not rest.startswith('#')):
            raise ValueError(f'{where}: unbalanced quotes')
        return raw[1:end]
    return re.split(r'\s+#', raw, maxsplit=1)[0].strip()


def read_site(path):
    values = {}
    for n, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if not sep or not re.fullmatch(r'[A-Z][A-Z0-9_]*', key.strip()):
            raise ValueError(f'{path}:{n}: expected KEY=VALUE')
        values[key.strip()] = site_value(value, f'{path}:{n}')
    return values


def rank_values(site, rank, ranks):
    out = dict(site)
    ips = [x.strip() for x in site.get('SWITCHLESS_RANK_HOST_IPS', '').split(',') if x.strip()]
    explicit = site.get(f'SWITCHLESS_RANK{rank}_HOST_IP')
    if explicit:
        out['SWITCHLESS_RANK_HOST_IP'] = explicit
    elif ips:
        if len(ips) != ranks:
            raise ValueError(f'SWITCHLESS_RANK_HOST_IPS lists {len(ips)} addresses; this profile has {ranks} ranks')
        out['SWITCHLESS_RANK_HOST_IP'] = ips[int(rank)]
    for key, root in (('SWITCHLESS_JIT_CACHE_DIR', 'SWITCHLESS_JIT_CACHE_ROOT'),
                      ('SWITCHLESS_PROFILES_DIR', 'SWITCHLESS_PROFILES_ROOT')):
        explicit = site.get(key.replace('SWITCHLESS_', f'SWITCHLESS_RANK{rank}_', 1))
        if explicit:
            out[key] = explicit
        elif site.get(root):
            out[key] = site[root].rstrip('/') + f'/rank{rank}'
    name = site.get(f'SWITCHLESS_RANK{rank}_CONTAINER_NAME')
    out['SWITCHLESS_CONTAINER_NAME'] = name or f"{site.get('SWITCHLESS_CONTAINER_PREFIX', 'switchless')}-rank{rank}"
    out['RANK'] = str(rank)
    out['OVERLAY_ROOT'] = site.get('SWITCHLESS_OVERLAY_ROOT', '')
    return out


def render(profile, site, rank):
    profile = with_options(profile, site)
    ranks = profile['ranks']
    if not 0 <= int(rank) < ranks:
        raise ValueError(f'rank must be 0..{ranks - 1}')
    values = rank_values(site, str(rank), ranks)
    tokens = []
    for tok in profile['argv_template']:
        if tok == '${ROLE_ARGS}':
            tokens += profile['role_args']['head' if str(rank) == '0' else 'worker']
        else:
            tokens.append(tok)
    missing, bad = set(), []

    def substitute(tok, in_mount):
        def sub(m):
            key = m.group(1)
            value = values.get(key, '')
            if not value:
                missing.add(key)
            elif any(ord(c) < 32 or ord(c) == 127 for c in value):
                bad.append(f'{key} contains a control character')
            elif in_mount and ',' in value:
                bad.append(f'{key} is used in a --mount option and must not contain a comma: {value!r}')
            return value
        return PLACEHOLDER.sub(sub, tok)
    out = [substitute(tok, i > 0 and tokens[i - 1] == '--mount') for i, tok in enumerate(tokens)]
    image = site.get('SWITCHLESS_IMAGE')
    if image:  # the same image under a registry reference; the measured runs used the local image ID
        if not PINNED_IMAGE.fullmatch(image):
            bad.append(f'SWITCHLESS_IMAGE must be pinned by digest (NAME@sha256:<64 hex>), not {image!r}')
        out = [image if tok == profile['image']['id'] else tok for tok in out]
    if missing:
        raise ValueError('site file lacks: ' + ', '.join(sorted(missing)))
    if bad:
        raise ValueError('; '.join(bad))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True, choices=('tp4', 'tp2'))
    ap.add_argument('--site', required=True, type=Path)
    ap.add_argument('--format', default='sh', choices=('sh', 'json'))
    ap.add_argument('--rank', type=int)
    a = ap.parse_args()
    profile = json.loads((ROOT / 'launch/profiles' / a.profile / 'profile.json').read_text())
    site = read_site(a.site)
    ranks = [a.rank] if a.rank is not None else list(range(profile['ranks']))
    try:
        argv = {str(r): render(profile, site, r) for r in ranks}
    except ValueError as exc:
        print('render: ' + str(exc), file=sys.stderr)
        return 2
    if a.format == 'json':
        print(json.dumps(argv, indent=1))
    else:
        print('# Rendered by launch/render.py; review before running. Start the worker ranks before rank 0.')
        for r in sorted(argv, key=int, reverse=True):
            print(f'# rank {r}')
            print(' '.join(shlex.quote(t) for t in argv[r]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
