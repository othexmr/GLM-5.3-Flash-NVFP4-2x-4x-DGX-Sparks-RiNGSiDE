# SPDX-License-Identifier: Apache-2.0
"""Compare the served files of a profile image with the served bytes recorded in sources/installed-files.json.

    python3 -B check_served.py --profile tp4 --manifest sources/installed-files.json [--root /] [--rank N] [--json]

Inside a profile image built by docker/Dockerfile (the recipe is copied to /opt/glm53-switchless/recipe):

    docker run --rm --entrypoint python3 IMAGE -B /opt/glm53-switchless/recipe/docker/tools/check_served.py \\
        --profile tp4 --manifest /opt/glm53-switchless/recipe/sources/installed-files.json

Every served file is classified:
- `measured`: the sha256 of the measured bytes (patched and repository files always are, or the build failed);
- `rebuilt`: a native artefact or third-party package file whose bytes differ from the measured build (expected: the
  builds are not bit-reproducible; docker/README.md says how to qualify them);
- `different` or `missing`: a source, patch or template file that differs; exit 1.
--rank selects the measured per-rank bytes of the prefill RDMA library (the measured nodes served two builds).
"""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE_KINDS = {'patch', 'src', 'template'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def check(profile, manifest, root, rank=None):
    entry = json.loads(Path(manifest).read_text())['profiles'][profile]
    rows = []
    for item in entry['files']:
        base = Path(root) / item['target'].lstrip('/')
        if item['kind'] in SOURCE_KINDS or item['kind'] == 'binary':
            files = {'': {'sha256': item['sha256']}}
        else:
            files = item['files']
        for name, f in files.items():
            path = base / name if name else base
            want = {f['sha256']}
            if f.get('per_rank_sha256'):
                want = {f['per_rank_sha256'][str(rank)]} if rank is not None else set(f['per_rank_sha256'].values())
            got = digest(path)
            binary = item['kind'] in ('binary', 'binary-dir', 'third-party-dir') or f.get('artefact')
            if got in want:
                status = 'measured'
            elif got is None:
                status = 'missing'
            elif binary:
                status = 'rebuilt'
            else:
                status = 'different'
            rows.append({'path': str(Path(item['target']) / name) if name else item['target'], 'kind': item['kind'],
                         'status': status, 'sha256': got})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True, choices=('tp4', 'tp2'))
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--root', type=Path, default=Path('/'))
    ap.add_argument('--rank', type=int)
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    rows = check(a.profile, a.manifest, a.root, a.rank)
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    if a.json:
        print(json.dumps({'profile': a.profile, 'counts': counts, 'files': rows}, indent=1))
    else:
        for r in rows:
            if r['status'] != 'measured':
                print(f"{r['status']:9} {r['path']}")
        print(f'check_served {a.profile}: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))
    return 1 if counts.get('different') or counts.get('missing') else 0


if __name__ == '__main__':
    raise SystemExit(main())
