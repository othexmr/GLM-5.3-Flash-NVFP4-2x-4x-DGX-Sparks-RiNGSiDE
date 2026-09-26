# SPDX-License-Identifier: Apache-2.0
"""Download the ten wheels of image stage 02 (b12x 1.3.0 and its CUDA Python and CUTLASS DSL set) by sha256.

    python3 -B fetch_wheels.py --sums wheels.SHA256SUMS --out DIR

Every file of wheels.SHA256SUMS is requested as name==version with its sha256 (`pip download --require-hashes`) and
the same platform options the lab used when it staged them (aarch64, CPython 3.12, binary only). DIR then holds the
ten wheels and SHA256SUMS, as the lab's build context of the stage did. A wheel whose bytes differ fails.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PLATFORM = ['--platform', 'manylinux2014_aarch64', '--platform', 'manylinux_2_28_aarch64',
            '--platform', 'manylinux_2_34_aarch64', '--python-version', '3.12', '--implementation', 'cp',
            '--abi', 'cp312', '--abi', 'none', '--only-binary=:all:']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sums', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    lines = [line.split() for line in a.sums.read_text().splitlines() if line.strip()]
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        req = Path(tmp) / 'requirements.txt'
        req.write_text(''.join(f"{name.split('-')[0]}=={name.split('-')[1]} --hash=sha256:{sha}\n"
                               for sha, name in lines))
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--require-hashes', '--no-cache-dir',
                        '--index-url', 'https://pypi.org/simple', *PLATFORM, '-r', str(req), '-d', str(a.out)],
                       check=True)
    shutil.copyfile(a.sums, a.out / 'SHA256SUMS')
    names = sorted(p.name for p in a.out.glob('*.whl'))
    expected = sorted(name for _, name in lines)
    if names != expected:
        raise SystemExit(f'fetch_wheels: downloaded {names}, expected {expected}')
    subprocess.run(['sha256sum', '-c', 'SHA256SUMS'], cwd=a.out, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
