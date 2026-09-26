# SPDX-License-Identifier: Apache-2.0
"""The repository licence, and the licence of the served chat template.

The served chat template (launch/profiles/*/chat_template_mm.jinja, docker/base/thinking-gate/chat_template_mm.jinja)
is MIT-licensed: it equals zai-org/GLM-5.3-Flash's chat_template.jinja at revision 690b705 with the thinking-gate hunk
of MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks eaf90d0 (Raymond Lucke, 2026-08-27, while that project was under the
MIT License), byte for byte. The inputs are in tests/cpu/data/chat-template (NOTICE has the history).
"""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'tests/cpu/data/chat-template'
ZAI_ORG = ('zai-org-GLM-5.3-Flash-690b705-chat_template.jinja',
           '0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5')
EAF90D0 = ('MiaAI-Lab-GLM-5.3-Flash-EXL3-2x-DGX-Sparks-eaf90d0-chat_template.jinja',
           '96ed83160b243de213e95eb2fa19bde4ac13b676661cfec477d18e45e9fcca3a')
HUNK = ('thinking-gate-eaf90d0.diff', '98e431892a72418322d0d7c52bfc7faa94b381a242e25ff2f98477d35cbbbe81')
SERVED = '7a5a0dda1331a7c40d930961cc1cb3b57c3b52625250c13372fe006ba2e9dfdb'
TEMPLATES = ('launch/profiles/tp4/chat_template_mm.jinja', 'launch/profiles/tp2/chat_template_mm.jinja',
             'docker/base/thinking-gate/chat_template_mm.jinja')
COMMAND = re.compile(r'(\d+)(?:,(\d+))?([acd])(\d+)(?:,(\d+))?')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def commands(diff):
    """The commands of a normal diff: (kind, old first, old last, new first, new last, removed lines, added lines)."""
    lines = diff.split(b'\n')
    if lines and lines[-1] == b'':
        lines.pop()
    out, i = [], 0
    while i < len(lines):
        m = COMMAND.fullmatch(lines[i].decode())
        assert m, lines[i]
        a1, a2, kind, b1, b2 = m.groups()
        a1, b1 = int(a1), int(b1)
        a2, b2 = int(a2 or a1), int(b2 or b1)
        i += 1
        removed, added = [], []
        while i < len(lines) and lines[i].startswith(b'< '):
            removed.append(lines[i][2:])
            i += 1
        if i < len(lines) and lines[i] == b'---':
            i += 1
        while i < len(lines) and lines[i].startswith(b'> '):
            added.append(lines[i][2:])
            i += 1
        out.append((kind, a1, a2, b1, b2, removed, added))
    return out


def apply(text, diff):
    """Apply a normal diff as `patch` does: each hunk's removed lines must occur once at or after the previous hunk, at
    the line the diff gives shifted by the offset found so far (the diff was made against an earlier file of the same
    template); an added block goes to the same shifted line."""
    old = text.split(b'\n')
    new, pos, offset = [], 0, 0   # pos: old lines consumed
    for kind, a1, a2, b1, b2, removed, added in commands(diff):
        if kind in 'cd':
            start = a1 - 1 + offset
            if old[start:start + len(removed)] != removed:
                found = [i for i in range(pos, len(old) - len(removed) + 1) if old[i:i + len(removed)] == removed]
                assert len(found) == 1, (kind, a1, 'removed lines found', len(found))
                start = found[0]
                offset = start - (a1 - 1)
            new += old[pos:start]
            pos = start + len(removed)
        else:
            keep = a1 + offset
            assert keep >= pos, (kind, a1, 'hunks out of order')
            new += old[pos:keep]
            pos = keep
        if kind in 'ac':
            assert len(added) == b2 - b1 + 1, (kind, a1, 'added line count')
            new += added
    new += old[pos:]
    return b'\n'.join(new)


class ChatTemplate(unittest.TestCase):
    def read(self, pinned):
        data = (DATA / pinned[0]).read_bytes()
        self.assertEqual(sha(data), pinned[1], pinned[0])
        return data

    def test_the_hunk_is_the_gate_of_eaf90d0(self):
        eaf = self.read(EAF90D0).split(b'\n')
        cmds = commands(self.read(HUNK))
        self.assertEqual([c[0] for c in cmds], ['a', 'c', 'c'])
        for kind, a1, a2, b1, b2, removed, added in cmds:
            self.assertEqual(eaf[b1 - 1:b2], added)
        self.assertTrue(any(b'thinking_enabled' in l for c in cmds for l in c[6]))

    def test_the_served_template_is_zai_orgs_with_the_hunk(self):
        rebuilt = apply(self.read(ZAI_ORG), self.read(HUNK))
        self.assertEqual(sha(rebuilt), SERVED)
        for rel in TEMPLATES:
            self.assertEqual((ROOT / rel).read_bytes(), rebuilt, rel)

    def test_the_thinking_contract_names_the_same_files(self):
        text = (ROOT / 'docker/base/thinking-gate/thinking_contract.py').read_text()
        self.assertIn(f'STOCK_TEMPLATE = "{ZAI_ORG[1]}"', text)
        self.assertIn(f'GATED_TEMPLATE = "{SERVED}"', text)

    def test_a_hunk_whose_lines_are_missing_is_refused(self):
        zai = self.read(ZAI_ORG)
        with self.assertRaises(AssertionError):
            apply(zai.replace(b"{{- '<think>' -}}", b"{{- '<think>' }}"), self.read(HUNK))


class RepositoryLicence(unittest.TestCase):
    def test_the_licence_is_the_apache_text(self):
        self.assertEqual((ROOT / 'LICENSE').read_bytes(), (ROOT / 'licenses/Apache-2.0.txt').read_bytes())
        comp = json.loads((ROOT / 'licenses/components.json').read_text())
        self.assertEqual(comp['original_code'], 'Apache-2.0')

    def test_the_chat_template_is_mit(self):
        comp = json.loads((ROOT / 'licenses/components.json').read_text())['components']
        for key in ('glm-template', 'miaai-lab-chat-template'):
            self.assertTrue(comp[key]['licence'].startswith('MIT'), key)
            self.assertTrue(set(TEMPLATES) <= set(comp[key]['used_by']), key)
        inst = json.loads((ROOT / 'sources/installed-files.json').read_text())
        for topo, prof in inst['profiles'].items():
            t = next(i for i in prof['files'] if i['kind'] == 'template')
            self.assertEqual(t['sha256'], SERVED)
            self.assertTrue(t['licence'].startswith('MIT ('), topo)


if __name__ == '__main__':
    unittest.main()
