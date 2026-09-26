# SPDX-License-Identifier: Apache-2.0
"""Bake a profile's overlay into the image at the served paths (docker/Dockerfile, stages profile-tp4 / profile-tp2).

    python3 -B bake_overlay.py --profile tp4 --overlay OVERLAY --manifest sources/installed-files.json [--root /]

OVERLAY is the output of sources/apply.py for the profile. For every served target of the profile
(sources/installed-files.json):
- a directory (kinds src-dir, config-dir, binary-dir, third-party-dir) replaces the image's directory as a whole, as
  the read-only bind mount of the measured launch hides the image's directory;
- a file (kinds patch, src, template, binary) replaces the image's file.
The measured launch mounted exactly these targets from an overlay directory (the `${OVERLAY_ROOT}` mounts of
launch/profiles/<profile>/profile.json); a baked image serves the same bytes at the same paths without those mounts.
Copies apply-receipt.json to /opt/glm53-switchless/<profile>/ and writes baked.json there. Exit 0 = every target baked.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

DIR_KINDS = {'src-dir', 'config-dir', 'binary-dir', 'third-party-dir'}
FILE_KINDS = {'patch', 'src', 'template', 'binary'}


def inside(base, target):
    rel = Path(target.lstrip('/'))
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('unsafe target ' + target)
    return Path(base) / rel


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def bake(profile, overlay, manifest, root):
    entry = json.loads(Path(manifest).read_text())['profiles'][profile]
    baked, problems = [], []
    for item in entry['files']:
        target, kind = item['target'], item['kind']
        src, dst = inside(overlay, target), inside(root, target)
        if kind in DIR_KINDS:
            if not src.is_dir():
                problems.append(f'{target}: directory missing from the overlay')
                continue
            remove(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, symlinks=True)
        elif kind in FILE_KINDS:
            if not src.is_file():
                problems.append(f'{target}: file missing from the overlay')
                continue
            if dst.is_dir() and not dst.is_symlink():
                problems.append(f'{target}: the image has a directory where the profile serves a file')
                continue
            remove(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        else:
            problems.append(f'{target}: unknown kind {kind}')
            continue
        baked.append(target)
    return baked, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True, choices=('tp4', 'tp2'))
    ap.add_argument('--overlay', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--root', type=Path, default=Path('/'))
    ap.add_argument('--record', type=Path, help='directory for apply-receipt.json and baked.json '
                    '(default: ROOT/opt/glm53-switchless/PROFILE)')
    a = ap.parse_args()
    baked, problems = bake(a.profile, a.overlay, a.manifest, a.root)
    record = a.record or inside(a.root, f'/opt/glm53-switchless/{a.profile}')
    record.mkdir(parents=True, exist_ok=True)
    receipt = a.overlay / 'apply-receipt.json'
    if receipt.is_file():
        shutil.copyfile(receipt, record / 'apply-receipt.json')
    (record / 'baked.json').write_text(json.dumps({'profile': a.profile, 'targets': baked, 'problems': problems},
                                                  indent=1) + '\n')
    if problems:
        print('bake_overlay: ' + '\n  '.join(problems), file=sys.stderr)
        return 1
    print(f'bake_overlay: {len(baked)} served targets of {a.profile} baked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
