# SPDX-License-Identifier: Apache-2.0
"""Prepare a pinned, experimental TP4 switched-RoCE option; never launch anything.

The measured profile stays unchanged. Like the existing weightless option, this
builds separate overlay files. Native compilation is explicit and is NOT a GPU
qualification. See docs/switched-roce.md before enabling this option.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SITE = 'SWITCHLESS_SWITCHED_ROCE_DIR'
FLAG = 'GLM53_RDMA_SWITCHED'
DP = '/usr/local/lib/python3.12/dist-packages/'


def changes():
    return json.loads((HERE / 'changes.json').read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def native_name():
    return 'libprb-' + changes()['glm53_prefill_rdma/_prb.cu']['sha256'][:16] + '.so'


def derive(root=ROOT):
    """Apply exact, generated edits to pinned inputs; fail before writing on drift."""
    result = {}
    for rel, entry in changes().items():
        data = (root / 'src/tp4' / rel).read_bytes()
        if sha(data) != entry['base_sha256']:
            raise ValueError('canonical source pin differs: ' + rel)
        lines = data.decode().splitlines(keepends=True)
        for edit in reversed(entry['edits']):
            start = edit['at']
            end = start + len(edit['before'])
            if lines[start:end] != edit['before']:
                raise ValueError('source anchor differs: ' + rel)
            lines[start:end] = edit['after']
        derived = ''.join(lines).encode()
        if sha(derived) != entry['sha256']:
            raise ValueError('derived source pin differs: ' + rel)
        result[rel] = derived
    return result


def with_profile(profile, site):
    """An explicit site option, not an import or a new qualification of the profile."""
    flag = site.get(FLAG, '0')
    if flag not in ('0', '1', ''):
        raise ValueError(FLAG + ' must be 0 or 1')
    if flag != '1':
        return profile
    if profile['topology'] != 'tp4' or not site.get(SITE):
        raise ValueError(FLAG + '=1 requires TP4 and a prepared ' + SITE)
    argv = list(profile['argv_template'])
    if FLAG + '=1' in argv:  # render and compose both call with_options
        return profile
    # These are direct-cable guards, not properties of the switched fabric.
    out, i = [], 0
    while i < len(argv):
        if argv[i] == '-e' and argv[i + 1].split('=', 1)[0] in (
                'NCCL_SWITCHLESS_RING_ONLY', 'NCCL_IB_ADDR_RANGE'):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    extra = ['-e', FLAG + '=1']
    files = list(changes()) + ['glm53_prefill_rdma/' + native_name()]
    for rel in files:
        extra += ['--mount', f'type=bind,src=${{{SITE}}}/{rel},dst={DP}{rel},readonly']
    image = out.index(profile['image']['id'])
    out[image:image] = extra
    return dict(profile, argv_template=out,
                site_variables=sorted(set(profile.get('site_variables', [])) | {SITE}))


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('refusing to replace different bytes: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--build-native', action='store_true',
                    help='compile the prefill SO with nvcc in the compatible ARM64 image; no GPU test')
    a = ap.parse_args(argv)
    try:
        files = derive()
        # Validate all existing output files before making any changes.
        for rel, data in files.items():
            dest = a.out / rel
            if dest.exists() and dest.read_bytes() != data:
                raise ValueError('refusing to replace different bytes: ' + str(dest))
        for rel, data in files.items():
            write_once(a.out / rel, data)
        so = a.out / 'glm53_prefill_rdma' / native_name()
        if so.exists() and not a.build_native:
            record = a.out / 'switched-roce-receipt.json'
            previous = json.loads(record.read_text()) if record.exists() else {}
            if (previous.get('files') != {k: sha(v) for k, v in files.items()}
                    or previous.get('native_sha256') != sha(so.read_bytes())):
                raise ValueError('native output has no matching build receipt')
        if a.build_native:
            if so.exists():
                raise ValueError('native output exists; use an empty output directory for a new build')
            temporary = so.with_suffix('.building.so')
            subprocess.run(['nvcc', '-O3', '-std=c++17', '-gencode', 'arch=compute_121,code=sm_121',
                            '-Xcompiler', '-fPIC', '-shared', '-o', str(temporary),
                            str(a.out / 'glm53_prefill_rdma/_prb.cu'), '-libverbs', '-lpthread'], check=True)
            os.replace(temporary, so)
        receipt = dict(option='experimental switched RoCE', qualification='CPU/source only; GPU/serving unqualified',
                       files={k: sha(v) for k, v in files.items()},
                       native_file='glm53_prefill_rdma/' + native_name(),
                       native_sha256=sha(so.read_bytes()) if so.exists() else None)
        # A build can add the native identity to an earlier source-only receipt.
        a.out.mkdir(parents=True, exist_ok=True)
        tmp = a.out / 'switched-roce-receipt.json.tmp'
        tmp.write_text(json.dumps(receipt, indent=2) + '\n')
        os.replace(tmp, a.out / 'switched-roce-receipt.json')
        print(json.dumps(receipt, indent=2))
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print('switched-roce: ' + str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
