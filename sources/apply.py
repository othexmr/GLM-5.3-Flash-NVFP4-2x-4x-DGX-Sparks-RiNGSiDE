# SPDX-License-Identifier: Apache-2.0
"""Build a profile's overlay directory (the files the launch mounts over the base image). Never contacts a node.

    python3 -B sources/apply.py --profile tp4 --image-root IMAGE_ROOT --out OVERLAY_ROOT \\
        [--artefacts DIR] [--third-party DIR] [--rank N] [--allow-rebuilt]

IMAGE_ROOT holds files copied out of the base image with their absolute paths below it, at least
IMAGE_ROOT/usr/local/lib/python3.12/dist-packages/{vllm,b12x,glm53_sparse_mla} (for example
`docker create IMAGE`, then `docker cp CONTAINER:/usr/local/lib/python3.12/dist-packages IMAGE_ROOT/usr/local/lib/python3.12/`).

For every served file of the profile (sources/installed-files.json):
- patch: the image file must have the recorded preimage sha256; the patch is applied with `git apply` (no fuzz,
  no three-way merge) and the result must have the served sha256;
- src/config/template: copied from this repository, sha256 checked;
- binary artefacts (NCCL library, sparse-MLA and FP8 GEMM extensions, prefill RDMA library): taken from
  --artefacts DIR (a file named by its sha256, or by its file name);
- the Humming package: taken from --third-party DIR, the directory the package was installed into
  (`pip install --target DIR`). Each served directory names its source directory below DIR explicitly
  (`source_dir` in the manifest): the TP4 profile mounts humming-kernels 0.1.15's `humming_kernels-0.1.15.dist-info`
  at the image's `humming_kernels-0.1.12.dist-info` path. A dist-info source must declare the recorded
  distribution name and version in its METADATA.
The sha256 of every artefact file is compared with the measured bytes. Rebuilt artefacts whose bytes differ are
refused unless --allow-rebuilt, and then reported as such.

Writes OVERLAY_ROOT/<absolute target path> and OVERLAY_ROOT/apply-receipt.json. Exit 0 only when every served
file is present and verified (or explicitly allowed as rebuilt).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_join(base, absolute):
    rel = Path(absolute.lstrip('/'))
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('unsafe target ' + absolute)
    return Path(base) / rel


def apply_patch(preimage, patch_bytes, rel):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(preimage)
        for extra in (['--check'], []):
            r = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-C', '3'] + extra, input=patch_bytes,
                               cwd=tmp, capture_output=True)
            if r.returncode:
                raise ValueError('git apply failed for ' + rel + ': ' + r.stderr.decode(errors='replace').strip())
        return target.read_bytes()


def canonical_name(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def third_party_base(third_party, item):
    """The directory below --third-party that supplies a third-party-dir entry, validated before any file is read."""
    source_dir = item.get('source_dir')
    if type(source_dir) is not str or not source_dir or '/' in source_dir or source_dir in ('.', '..'):
        raise ValueError('manifest entry lacks a valid source_dir: ' + item['target'])
    if not third_party:
        raise LookupError('--third-party not given for ' + item['target'])
    base = Path(third_party) / source_dir
    if not base.is_dir():
        raise LookupError(f'{source_dir} not found under --third-party (needed for {item["target"]})')
    dist = item.get('distribution')
    if dist:
        meta = base / 'METADATA'
        if not meta.is_file():
            raise LookupError(f'{source_dir}/METADATA missing')
        headers = {}
        for line in meta.read_text(encoding='utf-8', errors='replace').splitlines():
            if not line.strip():
                break
            key, sep, value = line.partition(':')
            if sep:
                headers.setdefault(key.strip().lower(), value.strip())
        if (canonical_name(headers.get('name', '')) != canonical_name(dist['name'])
                or headers.get('version') != dist['version']):
            raise ValueError(f"{source_dir} is {headers.get('name')} {headers.get('version')}, "
                             f"expected {dist['name']} {dist['version']} for {item['target']}")
    return base


def find_artefact(directory, sha, name):
    if not directory:
        return None
    for candidate in (Path(directory) / sha, Path(directory) / name):
        if candidate.is_file():
            return candidate
    return None


def build(profile, image_root, out, artefacts=None, third_party=None, rank=None, allow_rebuilt=False, root=ROOT):
    manifest = json.loads((root / 'sources/installed-files.json').read_text())
    prof = manifest['profiles'][profile]
    install_root = manifest['install_root']
    out = Path(out)
    receipt = {'profile': profile, 'image': prof['image'], 'files': {}, 'problems': []}

    def record(target, status, sha=None):
        receipt['files'][target] = {'status': status, 'sha256': sha}

    def write(target, data):
        dest = safe_join(out, target)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def fail(target, exc):
        receipt['problems'].append(str(exc))
        record(target, 'missing' if isinstance(exc, LookupError) else 'refused')

    def artefact_file(kind, target, name, f, sub, base=None):
        if f.get('source'):
            data = (root / f['source']).read_bytes()
            if sha256(data) != f['sha256']:
                raise ValueError('repository copy differs: ' + f['source'])
            write(sub, data)
            record(sub, 'copied', f['sha256'])
            return
        wanted = {f['sha256']}
        if f.get('per_rank_sha256'):
            wanted = ({f['per_rank_sha256'][str(rank)]} if rank is not None
                      else set(f['per_rank_sha256'].values()))
        if kind == 'third-party-dir':
            path = base / name if (base / name).is_file() else None
        else:
            path = next((p for p in (find_artefact(artefacts, w, name) for w in sorted(wanted)) if p), None)
        if path is None:
            raise LookupError('artefact not supplied: ' + sub)
        data = path.read_bytes()
        status = 'measured-artefact' if sha256(data) in wanted else 'rebuilt-artefact'
        if status == 'rebuilt-artefact' and not allow_rebuilt:
            raise ValueError(sub + ' differs from the measured bytes; pass --allow-rebuilt')
        write(sub, data)
        record(sub, status, sha256(data))

    for item in prof['files']:
        target, kind = item['target'], item['kind']
        try:
            if kind == 'patch':
                rel = target[len(install_root):]
                src = safe_join(image_root, target)
                if not src.is_file():
                    raise LookupError('image file missing under --image-root: ' + target)
                pre = src.read_bytes()
                if sha256(pre) != item['preimage_sha256']:
                    raise ValueError(f'image file {target} is not the recorded preimage '
                                     f'({sha256(pre)} != {item["preimage_sha256"]}): wrong base image?')
                post = apply_patch(pre, (root / item['patch']).read_bytes(), rel)
                if sha256(post) != item['sha256']:
                    raise ValueError('patched file differs from the served file: ' + target)
                write(target, post)
                record(target, 'patched', item['sha256'])
            elif kind in ('src', 'template'):
                data = (root / item['source']).read_bytes()
                if sha256(data) != item['sha256']:
                    raise ValueError('repository copy differs from the served file: ' + item['source'])
                write(target, data)
                record(target, 'copied', item['sha256'])
            elif kind == 'binary':
                artefact_file(kind, str(Path(target).parent), Path(target).name, {'sha256': item['sha256']}, target)
            elif kind in ('src-dir', 'config-dir', 'binary-dir', 'third-party-dir'):
                base = third_party_base(third_party, item) if kind == 'third-party-dir' else None
                for name, f in item['files'].items():
                    sub = target + '/' + name
                    try:
                        artefact_file(kind, target, name, f, sub, base)
                    except (LookupError, ValueError, OSError) as exc:
                        fail(sub, exc)
                for link, dest in item.get('symlinks', {}).items():
                    p = safe_join(out, target + '/' + link)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    if p.is_symlink() or p.exists():
                        p.unlink()
                    os.symlink(dest, p)
                    record(target + '/' + link, 'symlink')
            else:
                raise ValueError('unknown kind ' + kind)
        except (LookupError, ValueError, OSError) as exc:
            fail(target, exc)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'apply-receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--profile', required=True, choices=('tp4', 'tp2'))
    ap.add_argument('--image-root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--artefacts', type=Path)
    ap.add_argument('--third-party', type=Path)
    ap.add_argument('--rank', type=int, help='select the per-rank artefact where the measured bytes differ by rank')
    ap.add_argument('--allow-rebuilt', action='store_true')
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        print('apply: --out must be a new or empty directory', file=sys.stderr)
        return 2
    receipt = build(a.profile, a.image_root, a.out, a.artefacts, a.third_party, a.rank, a.allow_rebuilt)
    print(json.dumps({'profile': a.profile, 'files': len(receipt['files']), 'problems': receipt['problems']}, indent=1))
    return 1 if receipt['problems'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
