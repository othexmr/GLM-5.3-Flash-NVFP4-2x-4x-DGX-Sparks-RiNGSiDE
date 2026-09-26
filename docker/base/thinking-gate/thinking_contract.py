"""Verify the parser/template pair from bytes, without importing vLLM or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path

STOCK_TEMPLATE = "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5"
GATED_TEMPLATE = "7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb"
OLD_PARSER = "40f752a3f575db5978bff4080f0064bb3c83c52fa978fdea2f93aa3d43085cd9"
GATED_PARSER = "98193e12d9ae56c72c75865506088610a5c602d791bd6b62bf36e986fdc6a242"
PARSER = "vllm/parser/glm53_moe.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, template, expected="gated"):
    actual = {"parser_sha256": digest(root / PARSER), "template_sha256": digest(template)}
    pairs = {"stock": (OLD_PARSER, STOCK_TEMPLATE), "gated": (GATED_PARSER, GATED_TEMPLATE)}
    if expected not in pairs or tuple(actual.values()) != pairs[expected]:
        raise ValueError("Parser/template pair differs from the required " + expected + " contract: " + str(actual))
    return dict(actual, variant=expected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/usr/local/lib/python3.12/dist-packages"))
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--expected", choices=("stock", "gated"), default="gated")
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.template, a.expected), sort_keys=True))
