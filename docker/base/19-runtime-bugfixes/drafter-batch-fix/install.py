"""Install the reviewed XQA change only over the exact captured source closure."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=Path, default=Path(sysconfig.get_paths()['purelib']))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'source-manifest.json').read_text())
    # Validate the complete closure before writing any member.
    for item in manifest:
        src = root / 'runtime-candidate' / item['path']
        dst = args.site / item['path']
        if hashlib.sha256(src.read_bytes()).hexdigest() != item['after']:
            raise RuntimeError(f"candidate source drift: {src}")
        digest = hashlib.sha256(dst.read_bytes()).hexdigest() if dst.is_file() else None
        if digest not in (item['before'], item['after']):
            raise RuntimeError(f"base source drift: {dst}")
        if src.suffix == '.py':
            ast.parse(src.read_text(), filename=str(src))
    if not args.check:
        for item in manifest:
            dst = args.site / item['path']
            shutil.copyfile(root / 'runtime-candidate' / item['path'], dst)
    print('Verified XQA batch-invariant sliding source closure' if args.check else
          'Installed XQA batch-invariant sliding source closure; new JIT URI required')


if __name__ == '__main__':
    main()
