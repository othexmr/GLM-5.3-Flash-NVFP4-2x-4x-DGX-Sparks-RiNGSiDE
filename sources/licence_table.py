# SPDX-License-Identifier: Apache-2.0
"""Print (or --write into licenses/README.md) the per-file licence table generated from sources/installed-files.json."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = '## Every served file'


def table(root=ROOT):
    manifest = json.loads((root / 'sources/installed-files.json').read_text())
    prefix = manifest['install_root']
    merged = {}
    for topo, prof in manifest['profiles'].items():
        for item in prof['files']:
            if item['kind'] == 'binary':
                continue
            path = item['target'].replace(prefix, '') + ('/' if item['kind'].endswith('-dir') else '')
            key = (path, item['kind'], item['licence'], ', '.join(item.get('attribution') or []) or '-')
            merged.setdefault(key, set()).add(topo)
    lines = ['| served path | profiles | kind | licence | attribution keys (NOTICE, components.json) |', '|---|---|---|---|---|']
    for (path, kind, lic, attr), topos in sorted(merged.items()):
        lines.append(f'| `{path}` | {", ".join(sorted(topos))} | {kind} | {lic} | {attr} |')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    if not a.write:
        print(table(), end='')
        return 0
    readme = ROOT / 'licenses/README.md'
    text = readme.read_text()
    head, sep, rest = text.partition(MARKER)
    intro = rest.split('\n| ', 1)[0]
    readme.write_text(head + sep + intro + '\n' + table())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
