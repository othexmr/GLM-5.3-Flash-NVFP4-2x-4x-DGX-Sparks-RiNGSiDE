import hashlib
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/parser/glm53_moe.py")
if hashlib.sha256(p.read_bytes()).hexdigest() != "40f752a3f575db5978bff4080f0064bb3c83c52fa978fdea2f93aa3d43085cd9":
    raise RuntimeError("unexpected parser preimage")
