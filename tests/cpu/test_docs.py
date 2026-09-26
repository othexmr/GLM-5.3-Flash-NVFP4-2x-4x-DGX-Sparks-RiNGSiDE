# SPDX-License-Identifier: Apache-2.0
"""Documentation that must follow the profiles: configuration reference, licence table, results caveats."""
import importlib.util
import json
import re
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def env_values(topo):
    argv = json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())['argv_template']
    return dict(argv[i + 1].split('=', 1) for i in range(len(argv) - 1) if argv[i] == '-e')


def env_keys(topo):
    argv = json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())['argv_template']
    return {argv[i + 1].split('=', 1)[0] for i in range(len(argv) - 1) if argv[i] == '-e'}


class Docs(unittest.TestCase):
    def test_every_environment_variable_is_documented(self):
        doc = (ROOT / 'docs/configuration.md').read_text()
        for topo in ('tp4', 'tp2'):
            for key in env_keys(topo):
                self.assertIn(f'| `{key}` |', doc, (topo, key))

    def test_documented_values_are_the_profiles_values(self):
        """The TP4 and TP2 columns of docs/configuration.md give each profile's value ('-': not set; '(empty)': set
        to nothing; a value ending in '... (image value)' is an abbreviated image default)."""
        rows = {}
        for line in (ROOT / 'docs/configuration.md').read_text().splitlines():
            m = re.match(r'^\| `([A-Z][A-Z0-9_]*)` \| (.*?) \| (.*?) \| ', line)
            if m:
                rows[m.group(1)] = (m.group(2), m.group(3))
        for column, topo in enumerate(('tp4', 'tp2')):
            env = env_values(topo)
            for key, cells in rows.items():
                cell = cells[column]
                if key not in env:
                    self.assertEqual(cell, '-', (topo, key))
                elif cell == '(empty)':
                    self.assertEqual(env[key], '', (topo, key))
                elif cell.endswith('... (image value)`'):
                    self.assertTrue(env[key].startswith(cell[1:-len('... (image value)`')]), (topo, key))
                else:
                    self.assertEqual(cell, f'`{env[key]}`', (topo, key))

    def test_image_page_counts_the_patch_preimages_by_relation(self):
        lock = json.loads((ROOT / 'sources.lock.json').read_text())
        targets = {}
        for comp in ('vllm', 'b12x', 'sparse-mla'):
            for patch in lock['components'][comp]['patches']:
                relation = patch['preimage_upstream']
                kind = ('same' if relation.startswith('identical to vllm-project/vllm')
                        else 'port' if relation.startswith('file of the GLM-5.3 backport') else 'lab')
                targets.setdefault(kind, set()).add(patch['target'])
        page = (ROOT / 'docs/image.md').read_text()
        self.assertIn(f"| identical to vllm-project/vllm `98dff2a8` | {len(targets['same'])} |", page)
        self.assertIn(f"| GLM-5.3 backport file (absent from or different in `98dff2a8`) | {len(targets['port'])} |", page)
        self.assertIn(f"| earlier lab change baked into the image | {len(targets['lab'])} (", page)

    def test_readme_and_current_name_each_profiles_lab_plan(self):
        readme, current = (ROOT / 'README.md').read_text(), (ROOT / 'CURRENT.md').read_text()
        for topo in ('tp4', 'tp2'):
            plan = json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())['source_plan']['sha256']
            self.assertIn(f'`{plan[:8]}`', readme, topo)
            self.assertIn(f'`{plan}`', current, topo)

    def test_every_server_argument_is_documented(self):
        doc = (ROOT / 'docs/configuration.md').read_text()
        for topo in ('tp4', 'tp2'):
            p = json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())
            argv = p['argv_template'][p['argv_template'].index(p['image']['id']) + 1:] + p['role_args']['head']
            for tok in argv:
                if tok.startswith('--'):
                    self.assertIn(f'| `{tok}` |', doc, (topo, tok))

    def test_licence_table_matches_the_manifest(self):
        spec = importlib.util.spec_from_file_location('lt', ROOT / 'sources/licence_table.py')
        lt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lt)
        self.assertTrue((ROOT / 'licenses/README.md').read_text().endswith(lt.table()))

    def test_results_page_states_the_caveats(self):
        page = (ROOT / 'bench/results/README.md').read_text()
        for phrase in ('NON-DETERMINISTIC', 'single run', 'Room temperature', 'comparison recipes',
                       'at most 4', 'or 6', 'differs only in its header comment'):
            self.assertIn(phrase, page)
        self.assertNotRegex(page.lower(), r'\brivals?\b')

    def test_profiles_state_determinism(self):
        non = False
        for topo in ('tp4', 'tp2'):
            p = json.loads((ROOT / 'launch/profiles' / topo / 'profile.json').read_text())
            argv = ' '.join(p['argv_template'])
            atomic = 'B12X_GLM53_ATOMIC_PREFILL=1' in argv or 'B12X_GLM53_ATOMIC_DECODE_MIN_ROWS=' in argv
            if 'DETERMINISTIC BY CONSTRUCTION' in (p['nondeterministic_note'] or ''):
                # a checkpoint whose MoE kernels are deterministic by construction (W4A16): the atomic switches are
                # inert, and the claim is source-verified, not measured
                self.assertIsNone(p['deterministic'], topo)
                self.assertIn('not measured', p['nondeterministic_note'])
                continue
            self.assertEqual(p['deterministic'], False if atomic else None, topo)
            if atomic:
                non = True
                self.assertIn('NON-DETERMINISTIC', p['nondeterministic_note'])
        self.assertIn('NON-DETERMINISTIC' if non else 'not measured', (ROOT / 'README.md').read_text())


if __name__ == '__main__':
    unittest.main()
