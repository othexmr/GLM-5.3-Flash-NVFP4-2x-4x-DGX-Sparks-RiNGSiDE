"""Verify source/overlay hashes inside the candidate image, without CUDA imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("source", "overlay"), required=True)
    p.add_argument("--root", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages"))
    p.add_argument("--receipt", type=Path, default=Path("/opt/glm53-speedup/overlay-receipt.json"))
    a = p.parse_args()
    d = json.loads(a.receipt.read_text())
    for name, expected in d[a.stage].items():
        data = (a.root / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit("hash mismatch: " + name)
        if name.endswith(".py"):
            ast.parse(data, filename=name)
    print(a.stage + " hashes and Python syntax verified")
