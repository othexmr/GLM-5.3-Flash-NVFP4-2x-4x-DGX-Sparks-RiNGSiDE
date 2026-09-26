# SPDX-License-Identifier: Apache-2.0
"""The results page against its data: bench/results/README.md must be exactly what bench/results/build_page.py
renders from results.json and the evidence records, and the numbers derived on the page must follow from the raw
values in those records."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import unittest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'bench/results'
spec = importlib.util.spec_from_file_location('build_page', RESULTS / 'build_page.py')
build_page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_page)


def rows():
    return json.loads((RESULTS / 'results.json').read_text())


def evidence(name):
    return json.loads((RESULTS / build_page.EVIDENCE[name]).read_text())


def page():
    return (RESULTS / 'README.md').read_text()


def table_rows(section):
    """(recipe, configuration) of every row of the first table below a '## section' heading."""
    text = page().split('\n## ' + section + '\n', 1)[1]
    rows = []
    for line in text.split('\n| recipe |', 1)[1].splitlines()[2:]:
        if not line.startswith('| '):
            break
        cells = [c.strip() for c in line.strip('|').split(' | ')]
        rows.append((cells[0], cells[1]))
    return rows


class ResultsPage(unittest.TestCase):
    def test_page_is_rendered_from_the_data(self):
        self.assertEqual(page(), build_page.render(), 'run python3 -B bench/results/build_page.py --write')

    def test_every_evidence_record_exists_and_names_its_lab_windows(self):
        for name, rel in build_page.EVIDENCE.items():
            record = json.loads((RESULTS / rel).read_text())
            self.assertEqual(record['schema'], 'switchless-evidence/1', name)
            self.assertTrue(record['lab_receipts'], name)
        ids = [row['id'] for row in rows()]
        self.assertEqual(len(ids), len(set(ids)))
        for row in rows():
            self.assertTrue(row.get('lab_receipt'), row['id'])
        arms = evidence('arms')['rows']
        self.assertEqual([a['id'] for a in arms], ids)
        for a in arms:
            for field, value in a['fields'].items():
                self.assertIn(value['status'], ('recorded', 'recorded_withheld', 'not_recorded'), (a['id'], field))

    def test_mean_rank_order_recomputed_independently(self):
        rs = rows()
        groups = {'TP4': [r for r in rs if r['tp'] == 'TP4' and r['table'] == 'main'],
                  'TP2': [r for r in rs if r['tp'] == 'TP2' and r['table'] == 'main'],
                  'Additional measured arms': [r for r in rs if r['table'] == 'extra']}
        for section, group in groups.items():
            ranks = {r['id']: [] for r in group}
            for _, get, low in build_page.METRICS:
                values = sorted({get(r) for r in group if get(r) is not None}, reverse=not low)
                for r in group:
                    v = get(r)
                    if v is not None:
                        # rank = 1 + number of strictly better values, ties get the mean position
                        better = sum(1 for x in (get(o) for o in group) if x is not None and (x < v if low else x > v))
                        same = sum(1 for o in group if get(o) == v)
                        ranks[r['id']].append(better + (same + 1) / 2)
            expected = sorted(group, key=lambda r: statistics.mean(ranks[r['id']]))
            self.assertEqual(table_rows(section), [(r['recipe'], r['configuration']) for r in expected], section)

    def test_missing_cells_are_shown_and_left_out_of_the_mean(self):
        text = page()
        self.assertIn('A missing cell', text)
        for row in rows():
            n = sum(1 for _, get, _ in build_page.METRICS if get(row) is not None)
            self.assertIn(f'({n} of {len(build_page.METRICS)})', text, row['id'])

    def test_comparisons_are_whole_recipe_and_state_the_prompt_dependence(self):
        text = page()
        self.assertIn('whole recipe', text)
        self.assertGreaterEqual(text.count('depend on the prompt text'), 2)
        self.assertNotRegex(text.lower(), r'\brivals?\b')

    def test_kda_comparison_follows_from_its_record(self):
        kda = evidence('kda')
        v = kda['values']
        cells = [c['change_pct'] for c in v['decode_step_time']['cells']]
        for c in v['decode_step_time']['cells']:   # each cell's change follows from its step times
            self.assertAlmostEqual(c['change_pct'], (c['cand_ms'] / c['ref_ms'] - 1) * 100, delta=0.011)
        text = page()
        self.assertIn(f'decode step time, {len(cells)} fixed (streams, K) cells', text)
        self.assertIn(f'median {statistics.median(cells):+.2f} %, range {min(cells):+.2f} to {max(cells):+.2f} %', text)
        acc = v['speculative_acceptance']
        self.assertIn(f"{acc['previous_base']['rate']:.4f} | {acc['final_profile']['rate']:.4f}", text)
        for side in ('previous_base', 'final_profile'):
            a = acc[side]
            self.assertAlmostEqual(a['rate'], a['accepted_tokens'] / a['proposed_tokens'], delta=0.00005)
        for n, cell in v['cold_ttft']['cells'].items():
            self.assertIn(f"{cell['previous_base_s']:.3f} s | {cell['final_profile_s']:.3f} s", text)
        self.assertFalse(kda['verdicts']['rule_fixed_1232']['rule_met'])
        self.assertIn('did not pass clause 4', text)

    def test_tp2_promotion_lists_every_failed_clause(self):
        promo = evidence('tp2')
        failed = [c['clause'] for c in promo['rule']['clauses'] if not c['passed']]
        self.assertEqual(failed, promo['promotion']['failed_clauses'])
        section = page().split('\n## TP2\n', 1)[1].split('\n## ', 1)[0]
        for c in promo['rule']['clauses']:
            self.assertRegex(section, r'\| ' + re.escape(c['clause']) + r' \| .* \| ' + ('passed' if c['passed'] else r'\*\*failed\*\*') + r' \|')
        self.assertIn('owner promoted', page())

    def test_tp2_row_is_the_recorded_window(self):
        row = next(r for r in rows() if r['id'] == 'tp2-ringside')
        rec = evidence('tp2_row')['values']['cells']
        self.assertEqual(row['lab_receipt'], evidence('tp2_row')['lab_receipts'][0])
        comparison = evidence('tp2_row').get('information_comparison')
        if comparison:   # a TP2 row without a fresh reference: the lab's information comparison is on the page
            self.assertIn(comparison['line'], ' '.join(page().split()))
        for n in ('8192', '32768', '65536'):
            self.assertEqual(row['cold'][n], rec['cold_ttft_s'][n])
        for level, cell in rec['staggered'].items():
            self.assertEqual(row['staggered'][level], [cell['prefill_first_newcomer_ttft_s'], cell['decode_first_newcomer_ttft_s']])

    def test_temperature_caveat_follows_from_its_record(self):
        drift = evidence('drift')
        rep = next(r for r in drift['values']['same_plan_repeats'] if r.get('hours_apart', 0) >= 2)
        cells = [c['change_pct'] for c in rep['decode_step_time']['cells']]
        text = ' '.join(page().split())
        self.assertIn(f'differed by {statistics.median(cells):+.2f} % in median decode step time', text)
        self.assertIn(f'by {min(cells):+.2f} to {max(cells):+.2f} % in single cells', text)
        self.assertNotIn('about 1 to 2 %', text)

    def test_run_window_files_match_their_manifest_and_cards(self):
        for rel in build_page.RUNS:
            run = RESULTS / rel
            manifest = json.loads((run / 'manifest.json').read_text())
            for f in manifest['included']:
                data = (run / f['path']).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), f['sha256'], (rel, f['path']))
                self.assertEqual(len(data), f['bytes'], (rel, f['path']))
            # a card RigMark did not render is listed in the manifest with its reason (not_in_window)
            missing = {x['path']: x['why'] for x in manifest.get('not_in_window', [])}
            self.assertTrue(all(p.endswith('.card.txt') and why for p, why in missing.items()), (rel, missing))
            # the card names the receipt as RigMark wrote it: the included file, or its original when the lab's model
            # directory was replaced by the site placeholder (status redacted, original_sha256)
            entry = next(f for f in manifest['included'] if f['path'] == 'rigmark/code-full.json')
            code = entry.get('original_sha256') or hashlib.sha256((run / 'rigmark/code-full.json').read_bytes()).hexdigest()
            self.assertEqual(entry['status'] == 'redacted', 'original_sha256' in entry, (rel, entry['status']))
            if 'rigmark/code-full.card.txt' not in missing:
                card = (run / 'rigmark/code-full.card.txt').read_text()
                self.assertIn('sha256:' + code[:16], card)
            extract = json.loads((run / 'rigmark/prose-concurrency.rounds.json').read_text())['extract']
            if 'rigmark/prose-concurrency.card.txt' not in missing:
                card = (run / 'rigmark/prose-concurrency.card.txt').read_text()
                self.assertIn('sha256:' + extract['sha256'][:16], card)
            omitted = {o['window_path']: o for o in manifest['omitted']}
            self.assertEqual(omitted['native/prose-concurrency.json']['sha256'], extract['sha256'])
            self.assertEqual(manifest['window'], rel.split('/', 1)[1])

    def check_row_is_the_median_of_its_runs(self, row_id, rel, n_runs):
        run = RESULTS / rel
        code = json.loads((run / 'rigmark/code-full.json').read_text())
        prose = json.loads((run / 'rigmark/prose-concurrency.rounds.json').read_text())
        row = next(r for r in rows() if r['id'] == row_id)
        self.assertEqual(row['lab_receipt'], rel.split('/', 1)[1])
        self.assertEqual((row['runs'], code['settings']['runs'], prose['settings']['staggered_runs']), (n_runs,) * 3)
        for n in ('8192', '32768', '65536'):
            self.assertEqual(row['cold'][n], code['prefill'][n]['cold']['ttft_seconds']['median'])
            self.assertEqual(row['replay'][n], code['prefill'][n]['warm_replay']['ttft_seconds']['median'])
        for k in ('code', 'prose', 'structured'):
            self.assertEqual(row['decode'][k], code['decode'][k]['decode_tokens_per_second']['median'])
        for n, v in code['concurrency'].items():
            self.assertEqual(row['agg'][n], v['aggregate_end_to_end_tokens_per_second']['median'])
            self.assertEqual(len(v['rounds']), n_runs)
        for n, v in prose['concurrency'].items():
            self.assertEqual(row['prose_agg'][n], v['aggregate_end_to_end_tokens_per_second']['median'])
        for level, v in prose['staggered'].items():
            self.assertEqual(row['staggered'][level], [v['prefill_first']['newcomer_median_ttft_seconds']['median'],
                                                       v['decode_first']['newcomer_ttft_seconds']['median']])
            self.assertEqual((v['prefill_first']['valid_rounds'], v['decode_first']['valid_rounds']), (n_runs,) * 2, level)
        for n, v in code['prefill'].items() if 'prefill_suite' in row else ():
            self.assertEqual(row['prefill_suite'][n]['cold_s'], v['cold']['ttft_seconds']['median'])
            self.assertEqual(row['prefill_suite'][n]['replay_tok_s'], v['warm_replay']['effective_prefill_tokens_per_second']['median'])

    def test_tp4_row_is_the_median_of_its_runs(self):
        """The RiNGSiDE TP4 row: the release window when its record exists (then the final TP4 run of 387b3ff3 is the
        additional arm tp4-previous-run, with its 2 runs), else the final TP4 run."""
        row = next(r for r in rows() if r['id'] == 'tp4-ringside')
        self.check_row_is_the_median_of_its_runs('tp4-ringside', build_page.RUN_TP4, row['runs'])
        self.assertIn('262,136', page())
        if build_page.release_record():
            self.assertEqual(build_page.RUN_TP4, 'runs/' + build_page.release_record()['window'])
            self.check_row_is_the_median_of_its_runs('tp4-previous-run', build_page.RUN_PREVIOUS, 2)
        else:
            self.assertEqual(row['runs'], 2)

    def test_release_window_record(self):
        rel = build_page.release_record()
        if not rel:
            self.skipTest('no release window yet')
        row = next(r for r in rows() if r['id'] == 'tp4-ringside')
        profile = json.loads((ROOT / 'launch/profiles/tp4/profile.json').read_text())['source_plan']
        self.assertEqual(rel['plan_sha256'], profile['sha256'])
        self.assertEqual(row['lab_plan'], profile['sha256'][:8])
        self.assertEqual(row['lab_receipt'], rel['window'])
        self.assertEqual(row.get('sparkdash_window'), rel['window'])
        ko = rel['correctness']['korean']
        text = ' '.join(page().split())
        self.assertIn(f"{ko['runs'] - len(ko['breaks'])} of {ko['runs']} runs complete without a long-generation break", text)
        for cond, c in (rel['correctness'].get('emoji') or {}).get('conditions', {}).items():   # absent: not run
            self.assertIn(f"| emoji / CJK wall, {cond} | {c['n']} requests, {c['tokens']:,} tokens, thinking off | "
                          f"{c['fffd_total']} U+FFFD", page())
        te = evidence('tool_eval')
        self.assertIn(rel['window'], [r['window'] for r in te['runs']])

    def test_previous_tp4_profile_row_is_the_median_of_its_5_runs(self):
        self.check_row_is_the_median_of_its_runs('tp4-previous-profile', build_page.RUN5, 5)

    def test_tool_eval_table_follows_its_record(self):
        te = evidence('tool_eval')
        text = page()
        section = text.split('\n### tool-eval-bench (TP4)\n', 1)[1].split('\n## ', 1)[0]
        lines = [line for line in section.splitlines() if line.startswith('| ') and not line.startswith('| recipe')]
        self.assertEqual(len(lines), len(te['runs']))
        self.assertTrue(lines[0].startswith('| ' + build_page.SWITCHLESS + ' |'))   # RiNGSiDE first
        for r in te['runs']:
            self.assertEqual(r['counts']['passed'] * 2 + r['counts']['partial'], r['points'], r['window'])
            self.assertIn(f"| {r['final_score']} | {r['points']} / {r['max_points']} | {r['counts']['passed']} / "
                          f"{r['counts']['partial']} / {r['counts']['failed']} | {', '.join(r['failed_scenarios'])} | "
                          f"{r['median_turn_s']:.1f} | 1 |", section)

    def test_llama_benchy_table_follows_its_results(self):
        run = RESULTS / build_page.RUN_LLAMA_BENCHY   # the release plan's run on the NVIDIA checkpoint when llama-benchy
        summary = json.loads((run / 'llama-benchy/summary.json').read_text())   # did not run on the profile's plan
        runs = {}
        for name in ('passA.json', 'passB.json'):
            for b in json.loads((run / 'llama-benchy' / name).read_text())['benchmarks']:
                runs[(b['context_size'], b['concurrency'])] = runs.get((b['context_size'], b['concurrency']), []) + [b]
        text = page()
        for r in summary['rows']:
            tg = [b for b in runs[(r['depth'], r['concurrency'])] if b.get('tg_throughput')]
            self.assertTrue(any(abs(statistics.median(b['tg_throughput']['values']) - r['tg']) < 1e-6 for b in tg), r)
            self.assertIn(f"| {r['depth']:,} | {r['concurrency']} | {r['pp']:,.1f} | {r['tg']:.1f} |", text)

    def test_benchmark_fork_is_cited_by_its_public_commit(self):
        text = page()
        self.assertIn('40fabcaf', text)
        self.assertNotIn('not yet on GitHub', text)


if __name__ == '__main__':
    unittest.main()
