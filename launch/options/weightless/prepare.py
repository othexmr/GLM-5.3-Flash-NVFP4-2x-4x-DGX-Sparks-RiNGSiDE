# SPDX-License-Identifier: Apache-2.0
"""Prepare the optional weightless GLP-44 steering of the TP4 profile (off by default). Standard library only.

    python3 -B launch/options/weightless/prepare.py --out DIR [--cache DIR] [--offline] [--skip-vector]

msuiche/weightless (https://github.com/msuiche/weightless) steers the model with a projective activation vector,
GLP-44 at alpha 2.0 after every layer. Its repository states no licence, so this recipe carries none of its code: this
script fetches the GLM-5.3 patcher of weightless at a pinned commit, checks its sha256, reads the patcher's anchor and
replacement strings without running it (a literal evaluation of its module-level assignments), and applies them on
this machine. The vector comes from Hugging Face at a pinned revision; the repository is gated, so each operator
requests access there and fetches it with their own token (HF_TOKEN, or the token file of `huggingface-cli login`).

The steps, each checked by sha256 (any difference stops the script and nothing is written):
1. the TP4 profile's served model.py (f826b431), rebuilt from this repository alone: the file that
   docker/base/v029-port/v029-glm53-local.patch creates (03143971), then patches/vllm/tp4/ (git apply, no fuzz);
2. the patcher's five replacements applied to it, the forward block anchored after the served file's DFlash2 capture
   (the drafter reads unsteered hidden states; the last layer contracts after steering);
3. the recipe's own edits (EDITS below): the fail-closed guards (pinned GGUF vector and its sha256, alpha 2.0, the
   GLM-5.3 shape, layers 1-44, finite non-zero directions, the post_layer hook, TP2 or TP4 without sequence
   parallelism), the mHC prefill-ownership fix (only layer 0 starts from the full rows) and the notice line. Removed
   lines are named by count and sha256, never by text; every added line is the recipe's own;
4. the result must be 8cbfea91, the steered model.py of the lab's verification window (78693488) with the author note
   of one comment left out (line 755; token-equal with comments stripped); the vector must be 0ccce6b7.

DIR receives model.py, control.gguf and weightless-receipt.json. Copy DIR to the same path on every node and set
SWITCHLESS_WEIGHTLESS_DIR=DIR in the TP4 site file: launch/render.py and launch/compose.py then mount DIR/model.py
over the served model.py and DIR/control.gguf at /opt/weightless/control.gguf, and set the four WEIGHTLESS_*
variables (docs/configuration.md). Every rank logs "weightless GLP steering active" at boot (docs/operations.md).

--cache DIR: a directory with local copies (named by file name or by sha256) of the patcher and the vector, used
before any download. --offline: never download. Tokens are read from the environment or the token file only, sent
only to huggingface.co (not to the storage host it redirects to), and never printed or written.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[3]

UPSTREAM = dict(
    repository='msuiche/weightless',
    commit='15ed1373f6265c336ef227e7b4280e6ebdd347b3',
    path='patches/hotfix-glm53-steering-projective.py',
    sha256='63520932448ab51722888af3f97e4b8291c4d7ee6f923f48339a41a4fe05466d',
    bytes=35025,
)
UPSTREAM_URL = ('https://raw.githubusercontent.com/' + UPSTREAM['repository'] + '/' + UPSTREAM['commit'] + '/'
                + UPSTREAM['path'])
PATCH_NAMES = ('import os', 'module steering block', '__init__ buffers + _load_steering', 'forward apply',
               'last-layer deferral')
FORWARD = 'forward apply'
# A line of the served model.py (the end of its DFlash2 auxiliary hidden-state capture, in the decoder-layer loop).
FORWARD_ANCHOR = '                aux_hidden_states.append(aux_hidden_state)\n'

VECTOR = dict(
    repository='msuiche/GLM-5.3-Flash-abliterated-cyber-GLP-44',
    revision='ef85b0169fd25a3c0223eedddee124cf6b1580dc',
    file='GLM-5.3-Flash-abliterated-cyber-GLP-44-L1-44-a2.gguf',
    sha256='0ccce6b748f87da81505ff2bc3ca82429110940b6c0def60063d304785ccc00c',
    bytes=2887584,
    licence='MIT (model card)',
)
VECTOR_URL = ('https://huggingface.co/' + VECTOR['repository'] + '/resolve/' + VECTOR['revision'] + '/'
              + VECTOR['file'])

PORT_PATCH = 'docker/base/v029-port/v029-glm53-local.patch'
MODEL_REL = 'vllm/models/glm5next/nvidia/model.py'
PORT_SHA256 = '031439712aa46203846e1df03de95d7e38c42578c59c2f556d540ab09f26bf8c'
SERVED_PATCH = 'patches/vllm/tp4/models-glm5next-nvidia-model.py.patch'
SERVED_SHA256 = 'f826b4311615eb56c16dd1dea69cd0dbaf45df48f5d18dce05749a93b47d09c5'
UPSTREAM_APPLIED_SHA256 = 'b5936f470c2f5444d1b9b6039673eebd52b5f033f836eb343a7d09eab81d4eed'
STEERED_SHA256 = '8cbfea91c45de7f2f1f61ca4164f36647583bb812a3eecb1901accf432c27db7'
# the file the lab's verification window served (chain 348): the same bytes except one comment line, whose
# author note the recipe leaves out (token-equal code)
TESTED_SHA256 = '786934881d1c1f6b68ae71fda96b7a92818c39c13344dd2723632b0c0f5cd32e'
# the one line in which the two files differ: line 755, a comment whose author note the recipe leaves out; the served
# line is named by its sha256 (with its newline), the recipe's line is the first added line of the owner-row edit
TESTED_RELATION = dict(line=755, served_line_sha256='0dc3b67e46bd3f2f27aaa38631e78ab3a0a70dbbe77cd2bd838c000b47d75ccf',
                       same_code_without_comments=True)

MODEL_DST = '/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py'
VECTOR_DST = '/opt/weightless/control.gguf'
ENV = (('WEIGHTLESS_STEER_PATH', VECTOR_DST), ('WEIGHTLESS_STEER_ALPHA', '2.0'),
       ('WEIGHTLESS_STEER_HOOK', 'post_layer'), ('WEIGHTLESS_VECTOR_SHA256', VECTOR['sha256']))
MARKER = 'weightless GLP steering active'
FAILURE_MARKERS = ('Weightless vector identity mismatch', 'Weightless overlay requires its pinned GGUF vector')

# The recipe's edits of the patched file, as zero-context hunks: at = first line (1-based) in the patched file,
# remove = number of lines replaced, removed_sha256 = sha256 of those lines (with their newlines), add = new lines.
EDITS = (
    dict(at=4, remove=0, removed_sha256=None, add=(
        '# Modified by the GLM-5.3 RiNGSiDE recipe (othexmr): optional weightless GLP-44 steering (msuiche/weightless), weightless variant only.',
    )),
    dict(at=754, remove=1, removed_sha256='a165710ea94ae150df0c4cdebcbaf843c67361a3dfde1af053be571e8087d6b7', add=(
        '        # 2026-09-23 (GLP steering on the v8 owner): a steered layer hands the next one its post-layer stream on',
        "        # this rank's owner rows with no deferred state, so only layer 0 starts from the full rows (row count decides:",
        '        # owner rows = ceil(rows / 4) < rows).',
        '        first_full_pre = post is None and (',
        '            mhc_prefill_ownership is None or x.shape[0] == mhc_prefill_ownership.rows)',
    )),
    dict(at=767, remove=1, removed_sha256='7814632633c78996b538cb8e70c1ee7988b1030c2e1b0bae85e0a1bc8b1f2615', add=(
        '            if mhc_prefill_ownership is not None and first_full_pre:',
    )),
    dict(at=996, remove=0, removed_sha256=None, add=(
        '        # Scope pin widened 2026-09-19: the steering acts on the replicated post-layer mHC stream',
        '        # (hidden_size x mhc_num_residual_streams per rank, identical on every TP rank), so TP=4 on the',
        '        # four-Spark ring is the same computation as the qualified TP=2 path. Sequence parallel stays out.',
        '        if (vllm_config.parallel_config.tensor_parallel_size not in (2, 4)',
        '                or vllm_config.parallel_config.pipeline_parallel_size != 1',
        '                or self.is_sequence_parallel):',
        '            raise RuntimeError("Weightless profile requires TP2 or TP4, PP1, sequence parallel off")',
    )),
    dict(at=1041, remove=2, removed_sha256='c6a22c3ee1379546b2c8f87f5034c8c84bd63df08cb1bcdf3abe5ca15a60fa46', add=(
        '        if not path or not path.endswith(".gguf"):',
        '            raise RuntimeError("Weightless overlay requires its pinned GGUF vector")',
        '        import hashlib',
        '        from pathlib import Path',
        '        expected = os.environ.get("WEIGHTLESS_VECTOR_SHA256", "")',
        '        if len(expected) != 64 or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:',
        '            raise RuntimeError("Weightless vector identity mismatch")',
        '        if self._steer_alpha_val != 2.0:',
        '            raise RuntimeError("This GLP-44 profile is qualified only at alpha 2.0")',
        '        if (config.num_hidden_layers, config.hidden_size, config.mhc_num_residual_streams) != (45, 4096, 4):',
        '            raise RuntimeError("Weightless GLP-44 architecture mismatch")',
    )),
    dict(at=1044, remove=4, removed_sha256='cd6e31b50080a2fd96c318430ea016e370794206f95d2d4ab5ff6a85fac150d5', add=(
        '            raw = _load_gguf_control_vector(path)',
        '            if set(raw) != set(range(1, 45)):',
        '                raise RuntimeError("Weightless GLP-44 requires exact layers 1 through 44")',
        '            if any(not bool(torch.isfinite(v).all()) or not bool(torch.any(v != 0)) for v in raw.values()):',
        '                raise RuntimeError("Weightless vector has non-finite or zero directions")',
    )),
    dict(at=1052, remove=0, removed_sha256=None, add=(
        '            if selected is not None and selected != set(range(1, 45)):',
        '                raise RuntimeError("This Weightless profile requires all layers 1 through 44")',
        '            if _WEIGHTLESS_STEER_HOOK != "post_layer":',
        '                raise RuntimeError("This Weightless profile requires the post_layer hook")',
    )),
)


class Refused(Exception):
    """A pinned input or an intermediate result is not the recorded one."""


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def expect(data, want, what):
    got = sha256(data)
    if got != want:
        raise Refused(f'{what}: sha256 {got}, expected {want}')
    return data


def git_apply(tmp, patch, include=None):
    args = ['git', 'apply', '--whitespace=nowarn', '-C', '3'] + (['--include=' + include] if include else [])
    for extra in (['--check'], []):
        r = subprocess.run(args + extra + [str(patch)], cwd=tmp, capture_output=True)
        if r.returncode:
            raise Refused(f'git apply {patch.name}: ' + r.stderr.decode(errors='replace').strip())


def served_model(root=ROOT):
    """The TP4 profile's served model.py, rebuilt from this repository: the port file, then the profile patch."""
    with tempfile.TemporaryDirectory() as tmp:
        git_apply(tmp, root / PORT_PATCH, include=MODEL_REL)
        target = Path(tmp) / MODEL_REL
        expect(target.read_bytes(), PORT_SHA256, 'port model.py (' + PORT_PATCH + ')')
        git_apply(tmp, root / SERVED_PATCH)
        return expect(target.read_bytes(), SERVED_SHA256, 'served TP4 model.py (' + SERVED_PATCH + ')')


def literal_strings(source):
    """Module-level NAME = <string expression> assignments of a Python file, evaluated without running it: string
    literals, names bound earlier, `+` and tuples. Anything else is skipped."""
    values = {}

    def value(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.Tuple):
            return tuple(value(e) for e in node.elts)
        raise ValueError(type(node).__name__)
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = value(node.value)
            except ValueError:
                pass
    return values


def upstream_replacements(patcher):
    """The patcher's PATCHES (name, anchor, replacement) and FORWARD_BLOCK, checked for shape."""
    values = literal_strings(expect(patcher, UPSTREAM['sha256'], UPSTREAM['path']).decode('utf-8'))
    patches, block = values.get('PATCHES'), values.get('FORWARD_BLOCK')
    if (type(patches) is not tuple or tuple(p[0] for p in patches) != PATCH_NAMES
            or not all(len(p) == 3 and all(type(s) is str for s in p) for p in patches) or type(block) is not str):
        raise Refused('the patcher does not define the expected PATCHES and FORWARD_BLOCK')
    return patches, block


def apply_upstream(served, patches, block):
    text = served.decode('utf-8')
    for name, before, after in patches:
        if name == FORWARD:
            before, after = FORWARD_ANCHOR, FORWARD_ANCHOR + block + '\n'
        if text.count(before) != 1:
            raise Refused(f'anchor of {name!r} found {text.count(before)} times in the served model.py, expected once')
        text = text.replace(before, after, 1)
    return expect(text.encode('utf-8'), UPSTREAM_APPLIED_SHA256, 'served model.py with the patcher applied')


def apply_edits(data, edits=EDITS):
    lines = data.decode('utf-8').splitlines(True)
    for edit in sorted(edits, key=lambda e: e['at'], reverse=True):
        start = edit['at'] - 1
        removed = lines[start:start + edit['remove']]
        if len(removed) != edit['remove'] or start > len(lines):
            raise Refused(f"edit at line {edit['at']} is outside the file")
        if edit['remove'] and sha256(''.join(removed).encode('utf-8')) != edit['removed_sha256']:
            raise Refused(f"edit at line {edit['at']}: the lines it replaces are not the recorded ones")
        lines[start:start + edit['remove']] = [line + '\n' for line in edit['add']]
    return ''.join(lines).encode('utf-8')


def steered_model(patcher, root=ROOT):
    patches, block = upstream_replacements(patcher)
    patched = apply_upstream(served_model(root), patches, block)
    return expect(apply_edits(patched), STEERED_SHA256, 'steered model.py')


def from_cache(cache, names, want):
    if not cache:
        return None
    for name in names:
        path = Path(cache) / name
        if path.is_file():
            data = path.read_bytes()
            if sha256(data) == want:
                return data
    return None


class KeepAuthOnSameHost(urllib.request.HTTPRedirectHandler):
    """Follow redirects; the token goes along only when the redirect stays on huggingface.co."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        auth = req.unredirected_hdrs.get('Authorization')
        if new is not None and auth and urllib.parse.urlsplit(newurl).hostname == 'huggingface.co':
            new.add_unredirected_header('Authorization', auth)
        return new


def make_request(url, headers=None):
    request = urllib.request.Request(url, headers={'User-Agent': 'ringside-weightless-prepare'})
    for key, value in (headers or {}).items():
        request.add_unredirected_header(key, value)  # not forwarded to the host a redirect points to
    return request


def fetch(url, want, what, headers=None, timeout=60):
    try:
        opener = urllib.request.build_opener(KeepAuthOnSameHost)
        with opener.open(make_request(url, headers), timeout=timeout) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and 'huggingface.co' in url:
            raise Refused(f'{what}: HTTP {exc.code}. The repository is gated: request access on '
                          f'https://huggingface.co/{VECTOR["repository"]} with your account, then set HF_TOKEN '
                          'or run `huggingface-cli login`') from None
        raise Refused(f'{what}: HTTP {exc.code} from {url}') from None
    except (urllib.error.URLError, OSError) as exc:
        raise Refused(f'{what}: download failed ({exc})') from None
    return expect(data, want, what)


def hf_token():
    for key in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
        if os.environ.get(key, '').strip():
            return os.environ[key].strip()
    home = os.environ.get('HF_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    path = Path(os.environ.get('HF_TOKEN_PATH') or os.path.join(home, 'token'))
    if path.is_file():
        return path.read_text(encoding='utf-8').strip() or None
    return None


def get_patcher(cache, offline):
    data = from_cache(cache, (UPSTREAM['sha256'], Path(UPSTREAM['path']).name), UPSTREAM['sha256'])
    if data is None:
        if offline:
            raise Refused(f"the patcher is not in --cache and --offline is set: {UPSTREAM_URL}")
        data = fetch(UPSTREAM_URL, UPSTREAM['sha256'], UPSTREAM['path'])
    return data


def get_vector(cache, offline):
    data = from_cache(cache, (VECTOR['sha256'], VECTOR['file'], 'control.gguf'), VECTOR['sha256'])
    if data is None:
        if offline:
            raise Refused(f"the vector is not in --cache and --offline is set: {VECTOR_URL}")
        token = hf_token()
        if not token:
            raise Refused('no Hugging Face token: request access on https://huggingface.co/' + VECTOR['repository']
                          + ', then set HF_TOKEN or run `huggingface-cli login`')
        data = fetch(VECTOR_URL, VECTOR['sha256'], VECTOR['file'], headers={'Authorization': 'Bearer ' + token},
                     timeout=300)
    if len(data) != VECTOR['bytes']:
        raise Refused(f"{VECTOR['file']}: {len(data)} bytes, expected {VECTOR['bytes']}")
    return data


def write_once(path, data):
    """Write a file unless it already holds exactly these bytes; refuse to replace different bytes."""
    if path.exists():
        if path.read_bytes() != data:
            raise Refused(f'{path} exists with other content; choose an empty --out')
        return
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)


def receipt(vector_written):
    return dict(
        option='weightless GLP-44 steering (TP4 profile, off by default)',
        model=dict(path='model.py', sha256=STEERED_SHA256, mount=MODEL_DST, served_sha256=SERVED_SHA256,
                   tested_in_window_sha256=TESTED_SHA256, tested_relation=TESTED_RELATION,
                   port_sha256=PORT_SHA256, upstream_applied_sha256=UPSTREAM_APPLIED_SHA256),
        patcher=dict(UPSTREAM, url=UPSTREAM_URL, use='anchor and replacement strings read, not run'),
        vector=dict(VECTOR, path='control.gguf' if vector_written else None, mount=VECTOR_DST, url=VECTOR_URL),
        environment=[f'{k}={v}' for k, v in ENV],
        marker=MARKER, failure_markers=list(FAILURE_MARKERS),
        site='SWITCHLESS_WEIGHTLESS_DIR=<this directory, the same path on every node>',
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, type=Path, help='directory for model.py, control.gguf and the receipt')
    ap.add_argument('--cache', type=Path, default=os.environ.get('WEIGHTLESS_CACHE') or None,
                    help='local copies of the patcher and the vector (default: $WEIGHTLESS_CACHE)')
    ap.add_argument('--offline', action='store_true', help='never download; use --cache only')
    ap.add_argument('--skip-vector', action='store_true', help='build model.py only (the vector later)')
    a = ap.parse_args(argv)
    try:
        model = steered_model(get_patcher(a.cache, a.offline))
        vector = None if a.skip_vector else get_vector(a.cache, a.offline)
        a.out.mkdir(parents=True, exist_ok=True)
        write_once(a.out / 'model.py', model)
        if vector is not None:
            write_once(a.out / 'control.gguf', vector)
        write_once(a.out / 'weightless-receipt.json',
                   (json.dumps(receipt(vector is not None), indent=1) + '\n').encode('utf-8'))
    except Refused as exc:
        print('weightless: refused: ' + str(exc), file=sys.stderr)
        return 2
    out = a.out.resolve()
    print(f'model.py      sha256 {STEERED_SHA256}  {out / "model.py"}')
    if vector is not None:
        print(f'control.gguf  sha256 {VECTOR["sha256"]}  {out / "control.gguf"}')
    else:
        print('control.gguf  not fetched (--skip-vector); run again without it before enabling the option')
    print('Enable (TP4 only): copy this directory to the same path on every node and add to the TP4 site file')
    print(f'  SWITCHLESS_WEIGHTLESS_DIR={out}')
    print('launch/render.py and launch/compose.py then mount, per rank:')
    print(f'  {out}/model.py -> {MODEL_DST} (read-only)')
    print(f'  {out}/control.gguf -> {VECTOR_DST} (read-only)')
    print('  and set ' + ' '.join(f'{k}={v}' for k, v in ENV))
    print(f'Every rank logs "{MARKER}" at boot; remove the site line to turn the option off.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
