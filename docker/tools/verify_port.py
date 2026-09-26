# SPDX-License-Identifier: Apache-2.0
"""Verify the ported vLLM v0.29.0 source tree against the lab's records (docker/Dockerfile, stage src-vllm-v029).

    python3 -B verify_port.py --tree SRC --manifest source-port-manifest.json --files vllm-tree.sha256

SRC is vllm-project/vllm 98dff2a8 (Git tree d105bc35) with docker/base/v029-port/v029-glm53-local.patch applied.
Requires every changed file of the lab's source-port manifest (118 files, sha256 after the port, or absent where the
port deletes a file) and the complete vllm/ package tree the lab assembled into its wheel (vllm-tree.sha256: every
file, no extra and no missing one).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tree', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--files', required=True, type=Path)
    a = ap.parse_args()
    problems = []
    for change in json.loads(a.manifest.read_text())['changes']:
        path = a.tree / change['path']
        if change['sha256'] is None:
            if path.exists():
                problems.append('should be absent: ' + change['path'])
        elif not path.is_file() or sha(path) != change['sha256']:
            problems.append('differs from the lab port: ' + change['path'])
    listed = {}
    for line in a.files.read_text().splitlines():
        digest, name = line.split('  ', 1)
        listed[name] = digest
    present = {p.relative_to(a.tree).as_posix() for p in (a.tree / 'vllm').rglob('*')
               if p.is_file() and '__pycache__' not in p.parts}
    for name in sorted(present - set(listed)):
        problems.append('not in the lab tree: ' + name)
    for name, digest in sorted(listed.items()):
        if name not in present:
            problems.append('missing: ' + name)
        elif sha(a.tree / name) != digest:
            problems.append('differs from the lab tree: ' + name)
    if problems:
        print('verify_port: ' + '\n  '.join(problems[:50]), file=sys.stderr)
        return 1
    print(f'verify_port: {len(listed)} vllm/ files and the port manifest as recorded in docker/base/v029-port')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
