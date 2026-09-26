# SPDX-License-Identifier: Apache-2.0
"""No private paths, internal addresses or host names, retired identities, secrets or large binaries in the tree."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = {
    'private path': re.compile(rb'/Users/|/home/[a-z]'),
    'private address': re.compile(rb'\b(?:192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b|\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b'),
    'fabric address': re.compile(rb'\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    'lab host name': re.compile(rb'\bspark-0\d\b'),
    'secret': re.compile(rb'gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|xox[baprs]-'),
    'AI attribution': re.compile(rb'(?i)co-authored-by|generated with \[?claude|prepared with claude'),
}
# The fabric plan is a requirement of the measured NCCL build (fabric/README.md): only its four cable subnets may appear.
FABRIC_OK = re.compile(rb'\b10\.100\.22[4-7]\.0(?:/2[24])?\b')
SKIP_DIRS = {'.git', '__pycache__'}
# Words that must not appear anywhere, stored as sha256 of the lowercase word so this file does not repeat them:
# retired commit identities and hosts, and attributions the owner withdrew. A token is checked whole, split into its
# alphanumeric words, and (for hex words) by its first eight characters, so a commit id prefix matches too.
FORBIDDEN_WORDS = {
    'retired identity': {
        '7d92ba267803df4fd34f66e67bcc54a28429ce6c3284bd80e49f846caa2452b0',
        '4e12d9de33424d71edc358a85add5e001e05e44570ed9f292f16390cc90173a5',
        '699ccd441ef0ac1f73282c7ad6da5604a2ce8882d40f1d6e7e229d0a381f67a0',
        'dc7b87238c82b439441f51708bc38d60a1adb86bdf00035b45414ae6bb24ff98',
    },
    'earlier licence': {   # the repository licence is Apache-2.0 since 2026-09-26
        '54a2f35a8046ebff9056579231b93f59dd2058a8a2d2e75e25c73ab30d7a0bdd',
    },
    'withdrawn attribution': {
        '8b8616f1813d7f9ff027148ee4b2b6eb1209a503090c0e4633c76ab42c064d47',
        'cadc9c00dbe66e8de9f421b43d1b6025b6ca7550962e72bb2b908cedd103bf5a',
        '707a679aac35b386ecdbfb32fb081f773bd2c804759d9998df8fdc214b82e03f',
        '5a6e29f4fd5dd469db04350acd4499ddc75d5f7c6b9378b3fcd69d4782123666',
        '3ca8a1f8b199ee4c54453c46d008a92e981cbd38d64c61e38a8af339586c4c10',
    },
}
# Upstream vLLM's container user home, inside the Dockerfile that the lab's vLLM port patch adds; not a private path.
UPSTREAM_PATHS = {'docker/base/v029-port/v029-glm53-local.patch': b'/home/vllm'}
# Public text is written without "we", "our", "ours" and "us": documentation, notices, the site example and the recipe's
# own scripts. The lab's stage inputs (docker/base), served files (src/, patches/), licence texts, tests and build
# scripts keep their wording; lab window names such as 2026-09-24-final-replay-...-1 are quoted verbatim.
PERSONAL = re.compile(r'\b(we|our|ours|us)\b', re.I)
LAB_RECORD = re.compile(r'\b20\d\d-\d\d-\d\d-[a-z0-9]+(?:-[a-z0-9]+)*')
PUBLIC_CODE = ('docker/Dockerfile', 'launch/', 'docker/tools/', 'sources/')
TOKEN = re.compile(rb'[A-Za-z0-9_.+@-]+')
HEX = re.compile(rb'[0-9a-fA-F]{8,}')


def forbidden_words(data):
    found = []
    for m in TOKEN.finditer(data):
        token = m.group(0).lower()
        words = {token} | {w for w in re.split(rb'[^a-z0-9]+', token) if w}
        words |= {w[:8] for w in words if HEX.fullmatch(w)}
        for word in words:
            digest = hashlib.sha256(word).hexdigest()
            for label, hashes in FORBIDDEN_WORDS.items():
                if digest in hashes:
                    found.append((label, m.start()))
    return found


def files():
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
            continue
        yield rel, path


class Hygiene(unittest.TestCase):
    def test_no_forbidden_content(self):
        findings = []
        for rel, path in files():
            data = path.read_bytes()
            for label, pattern in PATTERNS.items():
                for m in pattern.finditer(data):
                    if label == 'fabric address' and FABRIC_OK.fullmatch(data[m.start():m.end() + 3].split(b' ')[0].rstrip(b',.;:|`)')) is not None:
                        continue
                    if label == 'fabric address' and FABRIC_OK.match(data, m.start()):
                        continue
                    if label == 'AI attribution' and rel.as_posix() == 'tests/cpu/test_hygiene.py':
                        continue
                    if label in ('private path', 'lab host name', 'private address') and rel.as_posix() == 'tests/cpu/test_hygiene.py':
                        continue
                    if label in ('fabric address', 'private address') and data[max(0, m.start() - 2):m.start()] == b'==':
                        continue  # a pinned version (name==10.x.y.z), not an address
                    allowed = UPSTREAM_PATHS.get(rel.as_posix())
                    if label == 'private path' and allowed and data.startswith(allowed, m.start()):
                        continue
                    findings.append(f'{rel}: {label}: {m.group(0)[:40]!r}')
            for label, offset in forbidden_words(data):
                findings.append(f'{rel}: {label} at byte {offset}')
        self.assertEqual(findings, [])

    def test_public_text_has_no_first_person_plural(self):
        findings = []
        for rel, path in files():
            name = rel.as_posix()
            if name.startswith(('src/', 'patches/', 'docker/base/', 'tests/', 'build/')) and name.endswith(('.py', '.cu')):
                continue
            if name.startswith(('src/', 'patches/', 'docker/base/')) or name.endswith('.txt'):
                continue
            if not (name.endswith('.md') or name in ('NOTICE', 'site.env.example')
                    or (name.startswith(PUBLIC_CODE) and name.endswith(('.py', '.sh', 'Dockerfile')))):
                continue
            for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                for m in PERSONAL.finditer(LAB_RECORD.sub('', line)):
                    findings.append(f'{name}:{n}: {m.group(0)!r}')
        self.assertEqual(findings, [])

    def test_first_person_plural_is_caught(self):
        self.assertTrue(PERSONAL.search('the profiles we serve'))
        self.assertTrue(PERSONAL.search('Our TP4 row'))
        self.assertFalse(PERSONAL.search(LAB_RECORD.sub('', 'window 2026-09-24-final-replay-ours-tp4-1')))
        self.assertFalse(PERSONAL.search('user, users, ourselves-free text'))

    def test_forbidden_words_are_caught(self):
        self.assertEqual(forbidden_words(b'pinned at ' + bytes.fromhex('6630363264306335') + b'fa8b'), [('withdrawn attribution', 10)])
        self.assertEqual(forbidden_words(b'nothing to see here, sha256 0123abcd'), [])

    def test_no_large_or_binary_files(self):
        for rel, path in files():
            self.assertLess(path.stat().st_size, 2_000_000, str(rel))
            head = path.read_bytes()[:4]
            self.assertNotEqual(head, b'\x7fELF', str(rel))
            self.assertNotIn(path.suffix, ('.so', '.whl', '.safetensors', '.tar', '.gz', '.pt', '.bin'), str(rel))

    def test_site_file_placeholders_are_documentation_addresses(self):
        text = (ROOT / 'site.env.example').read_text()
        for ip in re.findall(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', text):
            self.assertTrue(ip.startswith('192.0.2.') or ip.startswith('10.100.224.'), ip)


if __name__ == '__main__':
    unittest.main()
