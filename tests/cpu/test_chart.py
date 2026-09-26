# SPDX-License-Identifier: Apache-2.0
"""The README's results chart against its data: bench/results/chart-light.svg, chart-dark.svg and the <picture> block
between the chart markers in README.md must be exactly what bench/results/chart.py renders from results.json and the
evidence records, self-contained, and present in both themes."""
import importlib.util
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'bench/results'
spec = importlib.util.spec_from_file_location('chart', RESULTS / 'chart.py')
chart = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chart)
SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def svg(theme):
    return (RESULTS / chart.OUTPUTS[theme]).read_text()


class Chart(unittest.TestCase):
    def test_committed_files_match_the_data(self):
        self.assertEqual(chart.check(ROOT), [], 'run python3 bench/results/chart.py --write')

    def test_render_is_deterministic(self):
        for theme in chart.THEMES:
            self.assertEqual(chart.render_svg(theme), chart.render_svg(theme), theme)

    def test_svgs_are_self_contained(self):
        for theme in chart.THEMES:
            text = svg(theme)
            self.assertTrue(text.startswith('<svg ' + SVG_NS), theme)
            rest = text.replace(SVG_NS, '', 1)
            self.assertNotRegex(rest, r'(?i)https?:', theme)
            for token in ('@import', '@font-face', 'url(', '<image', 'href', '<script', '<foreignObject', 'data:'):
                self.assertNotIn(token, rest, (theme, token))

    def test_both_themes_present(self):
        light, dark = svg('light'), svg('dark')
        self.assertNotEqual(light, dark)
        self.assertIn(chart.THEMES['light']['surface'], light)
        self.assertIn(chart.THEMES['dark']['surface'], dark)
        self.assertNotIn(chart.THEMES['dark']['surface'], light)
        self.assertNotIn(chart.THEMES['light']['surface'], dark)
        readme = (ROOT / 'README.md').read_text()
        self.assertEqual(readme.count(chart.START), 1)
        self.assertEqual(readme.count(chart.END), 1)
        block = readme[readme.index(chart.START):readme.index(chart.END)]
        self.assertIn('<source media="(prefers-color-scheme: dark)" srcset="bench/results/chart-dark.svg">', block)
        self.assertIn('srcset="bench/results/chart-light.svg"', block)
        alt = re.search(r'<img alt="([^"]+)" src="bench/results/chart-light.svg"', block)
        self.assertIsNotNone(alt)
        self.assertIn('tool-eval-bench', alt.group(1))

    def test_values_come_from_the_rows(self):
        rows = {r['id']: r for r in json.loads((RESULTS / 'results.json').read_text())}
        light = svg('light')
        for row_id, _ in chart.SERIES:
            row = rows.get(row_id)
            if row is None:
                continue
            for prompt, value in row['decode'].items():
                self.assertIn(f'>{value:.1f}</text>', light, (row_id, prompt))
            for n, value in row['agg'].items():
                self.assertIn(f'>{value:.0f}</text>', light, (row_id, n))
            self.assertIn(f"lab plan {row['lab_plan']}", light)
        self.assertIn("a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol", light)
        tool = json.loads((RESULTS / 'evidence/tool-eval-bench.json').read_text())
        run = next(r for r in tool['runs'] if r['window'] == rows['tp4-ringside']['lab_receipt'])
        self.assertIn(f">{run['final_score']}</text>", light)
        self.assertIn(f">{run['points']} / {run['max_points']} points", light)   # ' each' with several rounds

    def test_only_this_recipes_rows(self):
        self.assertEqual([row_id for row_id, _ in chart.SERIES], ['tp4-ringside', 'tp2-ringside'])
        for theme in chart.THEMES:
            text = svg(theme)
            self.assertNotRegex(text.lower(), r'\brivals?\b')
            self.assertNotRegex(text, r'(?i)\b(we|our|ours|us)\b')


if __name__ == '__main__':
    unittest.main()
