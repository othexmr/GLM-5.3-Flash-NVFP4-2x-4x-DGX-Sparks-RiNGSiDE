# SPDX-License-Identifier: Apache-2.0
"""Pin the Python distributions of the vLLM source image to the set the lab's build installed.

    python3 -B pin_source.py --recorded distributions.txt [--mode exact|check|report] [--receipt FILE]

Runs inside the image built from ZJY0516/vllm 4500c80c's own docker/Dockerfile (docker/Dockerfile, stage `source`).
That upstream Dockerfile resolves several requirements at build time (version ranges, unpinned packages), so a later
build can install other releases than the lab's build of 2026-09-03 did: the lab's rebuild of 2026-09-08 already
installed newer releases of 15 packages. `distributions.txt` lists every distribution of that 2026-09-03 build
(name==version, /usr/local/lib/python3.12/dist-packages), the build the measured base image descends from.

Modes:
- exact (default): reinstall every drifted or missing release at the recorded version, without dependencies, from
  the indexes the upstream build used (PyPI, and for +cu130 builds the PyTorch or FlashInfer index); remove
  distributions the recorded build did not have; then require the recorded set exactly.
- check: change nothing and fail on any difference.
- report: change nothing, record the differences and continue (the result is then a different image).

Not pinned by this script: vllm itself (its local build; only the source commit 4500c80c is compared, the version
string also carries the build date), deep-ep (built from source by the upstream Dockerfile) and every package outside
Python (Ubuntu and CUDA packages installed with apt). Exit 0 = the recorded set (exact, check) or a written report.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import subprocess
import sys

SITE = Path('/usr/local/lib/python3.12/dist-packages')
LOCAL_BUILDS = {'vllm', 'deep-ep'}
INDEXES = {  # where the upstream build took the +cu130 local-version builds from
    'torch': 'https://download.pytorch.org/whl/cu130',
    'torchvision': 'https://download.pytorch.org/whl/cu130',
    'torchaudio': 'https://download.pytorch.org/whl/cu130',
    'torchcodec': 'https://download.pytorch.org/whl/cu130',
    'flashinfer-jit-cache': 'https://flashinfer.ai/whl/cu130',
}
PYPI = 'https://pypi.org/simple'
EXTRA = 'https://download.pytorch.org/whl/cu130'


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def recorded(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, sep, version = line.partition('==')
        if not sep:
            raise SystemExit(f'pin_source: bad line in {path}: {line!r}')
        out[canonical(name)] = version
    return out


def installed():
    out = {}
    for dist in metadata.distributions(path=[str(SITE)]):
        name = dist.metadata['Name']
        if name:
            out[canonical(name)] = dist.version
    return out


def compare(want, have):
    drift, missing, extra = {}, [], []
    for name, version in want.items():
        got = have.get(name)
        if got is None:
            missing.append(name)
        elif name == 'vllm':
            if '+g4500c80c' not in got:
                drift[name] = (version, got)
        elif got != version:
            drift[name] = (version, got)
    for name in have:
        if name not in want:
            extra.append(name)
    return drift, sorted(missing), sorted(extra)


def uv(*args):
    print('pin_source: uv ' + ' '.join(args), flush=True)
    subprocess.run(['uv', *args], check=True)


def repin(want, drift, missing, extra):
    names = sorted(set(drift) | set(missing))
    blocked = [n for n in names if n in LOCAL_BUILDS]
    if blocked:
        raise SystemExit('pin_source: cannot re-pin local builds ' + ', '.join(
            f'{n} (recorded {want[n]}, found {drift.get(n, (None, "missing"))[1]})' for n in blocked)
            + ': the upstream source build itself differs from the lab\'s; see docs/build-provenance.md')
    plain = [f'{n}=={want[n]}' for n in names if n not in INDEXES]
    if plain:
        uv('pip', 'install', '--system', '--no-deps', '--reinstall', '--index-url', PYPI, '--extra-index-url', EXTRA,
           '--index-strategy', 'unsafe-best-match', *plain)
    for n in names:
        if n in INDEXES:
            uv('pip', 'install', '--system', '--no-deps', '--reinstall', '--index-url', INDEXES[n], f'{n}=={want[n]}')
    if extra:
        uv('pip', 'uninstall', '--system', *extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--recorded', required=True, type=Path)
    ap.add_argument('--mode', choices=('exact', 'check', 'report'), default='exact')
    ap.add_argument('--receipt', type=Path, default=Path('/opt/glm53-switchless/source-distributions.json'))
    a = ap.parse_args()
    want = recorded(a.recorded)
    drift, missing, extra = compare(want, installed())
    receipt = {'mode': a.mode, 'recorded': str(a.recorded), 'before': {
        'drift': {n: {'recorded': w, 'found': g} for n, (w, g) in sorted(drift.items())},
        'missing': missing, 'extra': extra}}
    print(json.dumps(receipt['before'], indent=1), flush=True)
    if a.mode == 'exact' and (drift or missing or extra):
        repin(want, drift, missing, extra)
        drift, missing, extra = compare(want, installed())
    receipt['after'] = {'drift': {n: {'recorded': w, 'found': g} for n, (w, g) in sorted(drift.items())},
                        'missing': missing, 'extra': extra}
    receipt['matches_recorded_set'] = not (drift or missing or extra)
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    a.receipt.write_text(json.dumps(receipt, indent=1) + '\n')
    if receipt['matches_recorded_set'] or a.mode == 'report':
        print('pin_source: ' + ('the recorded distribution set' if receipt['matches_recorded_set']
                                else 'differences recorded in ' + str(a.receipt) + ' (mode report)'))
        return 0
    print('pin_source: the source image differs from the recorded distribution set: '
          + json.dumps(receipt['after']) + '\nRebuild with --build-arg SOURCE_PIN=exact, or accept a different image '
          'with SOURCE_PIN=report (it then needs its own qualification).', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
