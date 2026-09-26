# SPDX-License-Identifier: Apache-2.0
"""Check a rebuilt base image against what the recipe records about the measured one.

    python3 -B check_base.py --manifest sources/installed-files.json [--receipt FILE]

Runs inside the image (docker/Dockerfile, stage `base`). It requires:
- every file that a profile patch replaces to have the recorded preimage sha256 (`preimage_sha256` in
  sources/installed-files.json, both profiles), so sources/apply.py can apply the patches;
- the package versions docs/image.md lists for the measured base image.
The measured image itself is identified by its image ID only; a rebuild never has that ID, and files outside the
patch preimages are not compared here (docker/README.md, "Compare with the measured image").
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

VERSIONS = {'vllm': '0.29.0+glm53.local', 'torch': '2.13.0+cu130', 'transformers': '5.16.1',
            'flashinfer-python': '0.6.18', 'triton': '3.7.1', 'nvidia-cutlass-dsl': '4.6.2', 'b12x': '1.3.0'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--root', type=Path, default=Path('/'))
    ap.add_argument('--receipt', type=Path, default=Path('/opt/glm53-switchless/base-check.json'))
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    problems, checked = [], {}
    for profile, entry in manifest['profiles'].items():
        for item in entry['files']:
            if item['kind'] != 'patch':
                continue
            path = a.root / item['target'].lstrip('/')
            got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            checked[item['target']] = got
            if got != item['preimage_sha256']:
                problems.append(f"{item['target']}: {got or 'missing'} is not the recorded preimage "
                                f"{item['preimage_sha256']} ({profile})")
    versions = {}
    for name, want in VERSIONS.items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != want:
            problems.append(f'{name} {versions[name]} is not the recorded {want}')
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps({'preimages': checked, 'versions': versions, 'problems': problems}, indent=1) + '\n')
    if problems:
        print('check_base: the rebuilt base differs from the recorded base image:\n  ' + '\n  '.join(problems),
              file=sys.stderr)
        return 1
    print(f'check_base: {len(checked)} patch preimages and {len(versions)} package versions as recorded')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
