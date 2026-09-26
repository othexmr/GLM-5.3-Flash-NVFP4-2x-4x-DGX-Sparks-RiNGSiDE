# SPDX-License-Identifier: Apache-2.0
"""Render bench/results/README.md from results.json, the evidence records in evidence/ and the TP4 profile's source
plan (launch/profiles/tp4/profile.json: its lab plan and, for a derived plan, what it adds to the measured one).

    python3 -B bench/results/build_page.py [--check | --write]

Every number on the page comes from those JSON files; tests/cpu/test_results.py fails when the page and the data
disagree. The lab produces the JSON files from its receipts; this script only formats them.
"""
import argparse
import json
import re
from pathlib import Path
import statistics
import sys
import textwrap

HERE = Path(__file__).resolve().parent
SWITCHLESS = 'othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE'

# The ranked columns: (label, getter, lower is better). The replay matrix and the admission limit are not ranked.
METRICS = [
    ('cold TTFT 8K', lambda r: r['cold'].get('8192'), True),
    ('cold TTFT 32K', lambda r: r['cold'].get('32768'), True),
    ('cold TTFT 64K', lambda r: r['cold'].get('65536'), True),
    ('decode code', lambda r: r['decode'].get('code'), False),
    ('decode prose', lambda r: r['decode'].get('prose'), False),
    ('decode structured', lambda r: r['decode'].get('structured'), False),
    ('short code aggregate C1', lambda r: r['agg'].get('1'), False),
    ('short code aggregate C6', lambda r: r['agg'].get('6'), False),
] + [
    (f'staggered {mode} L{level}', lambda r, level=level, i=i: (r['staggered'].get(level) or [None, None])[i], True)
    for i, mode in enumerate(('prefill-first', 'decode-first')) for level in ('2', '4', '6')
] + [
    ('sparkDash code x1', lambda r: (r.get('sparkdash') or {}).get('codex1'), False),
]


EVIDENCE = {  # the records behind the page, in evidence/ (written by the lab from its receipts)
    'arms': 'evidence/comparison-arms.json',
    'kda': 'evidence/kda-checkpoints-tp4.json',
    'drift': 'evidence/temperature-drift.json',
    'tp2': 'evidence/tp2-weakspots-promotion.json',
    'tp2_row': 'evidence/tp2-row-69ad74fc.json',
    'tp2_previous_row': 'evidence/tp2-row-32dd2ab8.json',
    'replay': 'evidence/replay-matrix.json',
    'tool_eval': 'evidence/tool-eval-bench.json',
}
# The release window of the TP4 profile's own plan, when it has run (written by the lab from the window's receipts):
# the RiNGSiDE TP4 row, the correctness checks and the texts below follow it; without it the page shows the final TP4
# run of lab plan 387b3ff3 as the TP4 row.
RELEASE = 'evidence/release-window-tp4.json'


def release_record(root=HERE):
    path = root / RELEASE
    return json.loads(path.read_text()) if path.exists() else None


RUN_PREVIOUS = 'runs/2026-09-25-tp4-final-2'   # the final TP4 run of 387b3ff3: RigMark receipts (2 runs), llama-benchy
RUN_TP4 = ('runs/' + release_record()['window']) if release_record() else RUN_PREVIOUS   # the TP4 row's receipts
RUN5 = 'runs/2026-09-24-rigmark5-tp4-1'   # the previous TP4 profile's 5-round window: RigMark receipts and llama-benchy
# the release plan's RigMark run on the NVIDIA checkpoint, an additional arm when the profile serves another checkpoint
_NV = next((r for r in json.loads((HERE / 'results.json').read_text()) if r['id'] == 'tp4-release-nvidia'), None)
RUN_NVIDIA = ('runs/' + _NV['lab_receipt']) if _NV else None
RUNS = tuple(dict.fromkeys(x for x in (RUN_TP4, RUN_PREVIOUS, RUN5, RUN_NVIDIA) if x))
# The fast release set (the record names results that did not run on the plan, or a Korean check stopped early): the
# texts follow the record's order and not_run. When llama-benchy did not run on the plan, its section shows the release
# plan's llama-benchy run on the NVIDIA checkpoint (the record's nvidia_arm, copied into its run folder).
WORDS = {2: 'two', 3: 'three', 4: 'four', 5: 'five'}
NOT_RUN_TEXT = {'emoji': 'the emoji and CJK wall stress test', 'llama_benchy': 'llama-benchy'}
ORDER_TEXT = {'korean': 'the Korean check', 'emoji': 'the emoji and CJK wall stress test', 'rigmark': 'RigMark',
              'llama_benchy': 'llama-benchy', 'sparkdash': 'sparkDash', 'tool_eval': 'tool-eval-bench'}
_REL = release_record()
_NV_REC = (_REL or {}).get('nvidia_arm') or None
LB_NVIDIA = bool(_REL and 'llama_benchy' in (_REL.get('not_run') or []) and _NV_REC and _NV_REC.get('llama_benchy_shown'))
RUN_LLAMA_BENCHY = ('runs/' + _NV_REC['window']) if LB_NVIDIA else RUN_TP4   # the llama-benchy results shown


def fast(rel):
    """Whether the record is the fast release set: results stated as not run, or a Korean check stopped early."""
    return bool(rel and (rel.get('not_run') or rel['correctness']['korean'].get('stopped_early')))


def and_list(xs):
    xs = list(xs)
    return ', '.join(xs[:-1]) + (' and ' if len(xs) > 1 else '') + (xs[-1] if xs else '')


def not_run_sentence(rel):
    """'The emoji and CJK wall stress test and llama-benchy were not run on this plan.' from the record."""
    nr = [r for r in (rel.get('not_run') or []) if r in NOT_RUN_TEXT]
    if not nr:
        return ''
    t = and_list(NOT_RUN_TEXT[r] for r in nr)
    return f"{t[0].upper()}{t[1:]} {'was' if len(nr) == 1 else 'were'} not run on this plan"



FORK = {'repository': 'othexmr/rigmark', 'branch': 'feat/staggered-arrival',
        'commit': '40fabcaf6e963ac27ab347f61d54a225dbd07419', 'tree': '3ba4ae72a548bc72df66933163106829b0644e15',
        'measured': '6e0e24a6ff5061721b83a482d1967c6b4e07517f', 'parent': 'e748928d570e36274655a805debb1bd1265607ea'}


def load(root=HERE):
    rows = json.loads((root / 'results.json').read_text())
    evidence = {name: json.loads((root / rel).read_text()) for name, rel in EVIDENCE.items()}
    evidence['llama_benchy'] = json.loads((root / RUN_LLAMA_BENCHY / 'llama-benchy/summary.json').read_text())
    evidence['release'] = release_record(root)
    return {'rows': rows, 'evidence': EVIDENCE}, evidence


def f(v, d=2):
    return '-' if v is None else f'{v:.{d}f}'


def pct(a, b):
    return f'{(b / a - 1) * 100:+.2f} %'


def ranks(group):
    """Mean rank of each row over the METRICS it has a value for (1 = best; equal values share the mean of their
    positions). A missing cell is left out of that row's mean."""
    per_row = {id(r): [] for r in group}
    for _, get, low in METRICS:
        vals = [(get(r), r) for r in group if get(r) is not None]
        vals.sort(key=lambda x: x[0] if low else -x[0])
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[j + 1][0] == vals[i][0]:
                j += 1
            for k in range(i, j + 1):
                per_row[id(vals[k][1])].append((i + j) / 2 + 1)
            i = j + 1
    return {k: (sum(v) / len(v) if v else float('inf'), len(v)) for k, v in per_row.items()}


def ordered(group):
    r = ranks(group)
    return sorted(group, key=lambda row: r[id(row)][0])


def table(group):
    out = ['| recipe | configuration | admits | runs | cold TTFT 8K / 32K / 64K (s) | decode code / prose / structured '
           '(tok/s) | short code aggregate C1 / C6 (tok/s) | staggered prefill-first newcomer L2 / L4 / L6 (s) | '
           'staggered decode-first newcomer L2 / L4 / L6 (s) | sparkDash code x1 (tok/s) | replay matrix | mean rank |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    r = ranks(group)
    for row in ordered(group):
        st, rm = row['staggered'], row.get('replay_matrix')
        rep = 'not run'
        if rm:
            rep = ('valid' if rm['valid'] else 'not valid') + ', quality ' + rm['quality'] + ', ' + (
                f"meets the latency target up to {rm['capacity']} req/s" if rm['capacity'] is not None
                else 'no swept rate meets the latency target') + (f" (lab plan {row['replay_plan']})" if row.get('replay_plan') else '')
        sd = f((row.get('sparkdash') or {}).get('codex1'), 1) + (f" (lab plan {row['sparkdash_plan']})" if row.get('sparkdash_plan') else '')
        mean, n = r[id(row)]
        runs = str(row.get('runs', 1)) + (f" (staggered: {row['staggered_rounds']} rounds)"
                                          if row.get('staggered_rounds', 1) != row.get('runs', 1) else '')
        out.append(f"| {row['recipe']} | {row['configuration']} | {row['admits']} | {runs} | {f(row['cold'].get('8192'))} / "
                   f"{f(row['cold'].get('32768'))} / {f(row['cold'].get('65536'))} | {f(row['decode'].get('code'), 1)} / "
                   f"{f(row['decode'].get('prose'), 1)} / {f(row['decode'].get('structured'), 1)} | "
                   f"{f(row['agg'].get('1'), 1)} / {f(row['agg'].get('6'), 1)} | "
                   f"{' / '.join(f((st.get(l) or [None, None])[0]) for l in ('2', '4', '6'))} | "
                   f"{' / '.join(f((st.get(l) or [None, None])[1]) for l in ('2', '4', '6'))} | {sd} | {rep} | "
                   f"{mean:.2f} ({n} of {len(METRICS)}) |")
    return '\n'.join(out)


def prefill_table(tp4):
    lines = ['| prompt tokens | cold TTFT (s) | cold prefill (tok/s) | warm replay TTFT (s) | warm replay (tok/s) |',
             '|---:|---:|---:|---:|---:|']
    for n, c in sorted(tp4['prefill_suite'].items(), key=lambda kv: int(kv[0])):
        lines.append(f"| {int(n):,} | {c['cold_s']:.2f} | {c['cold_tok_s']:,.0f} | {c['replay_s']:.2f} | "
                     f"{c['replay_tok_s']:,.0f} |")
    return '\n'.join(lines)


def llama_benchy_table(summary):
    lines = ['| context depth (tokens) | concurrency | prompt processing (t/s) | generation (t/s) | per request: prompt / '
             'generation (t/s) | TTFT (ms) | runs |', '|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['rows']:
        lines.append(f"| {r['depth']:,} | {r['concurrency']} | {r['pp']:,.1f} | {r['tg']:.1f} | {r['pp_req']:,.1f} / "
                     f"{r['tg_req']:.1f} | {r['e2e_ttft_ms']:,.0f} | {r['runs']} |")
    return '\n'.join(lines)


def tool_eval_table(te):
    lines = ['| recipe | run | score | points | pass / partial / fail | failed scenarios | median turn (s) | runs |',
             '|---|---|---:|---:|---|---|---:|---:|']
    for r in sorted(te['runs'], key=lambda r: not r['recipe'].startswith(SWITCHLESS)):   # RiNGSiDE first
        lines.append(f"| {r['recipe']} | {r['what']} | {r['final_score']} | {r['points']} / {r['max_points']} | "
                     f"{r['counts']['passed']} / {r['counts']['partial']} / {r['counts']['failed']} | "
                     f"{', '.join(r['failed_scenarios'])} | {r['median_turn_s']:.1f} | 1 |")
    return '\n'.join(lines)


def many_streams(data):
    tp4 = next(r for r in data['rows'] if r['id'] == 'tp4-ringside')
    others = [r for r in data['rows'] if r['tp'] == 'TP4' and r['table'] == 'main' and r is not tp4]
    lines = ['| streams | short code aggregate (tok/s) | prose aggregate (tok/s) | staggered prefill-first newcomer (s) | '
             'staggered decode-first newcomer (s) |', '|---|---|---|---|---|']
    for n in ('8', '12', '16'):
        st = tp4['staggered'].get(n) or [None, None]
        lines.append(f"| {n} | {f(tp4['agg'].get(n), 1)} | {f(tp4['prose_agg'].get(n), 1)} | {f(st[0])} | {f(st[1])} |")
    caps = '; '.join(f"{r['recipe']} admits at most {r['admits']} concurrent requests"
                     for r in sorted(others, key=lambda r: r['admits']))
    return '\n'.join(lines), caps



def wrap(text, indent=''):
    """Wrap one paragraph (or bullet) at 118 columns, as the rest of the page."""
    return textwrap.fill(text, width=118, subsequent_indent=indent, break_long_words=False, break_on_hyphens=False)


def mean_pct(cells):
    xs = [c['change_pct'] for c in cells]
    return statistics.median(xs), min(xs), max(xs), len(xs)


def kda_section(kda):
    v = kda['values']
    cold, rep, steps, acc = v['cold_ttft'], v['warm_replay_ttft'], v['decode_step_time'], v['speculative_acceptance']
    ns = ('8192', '32768', '65536')
    med, lo, hi, n = mean_pct(steps['cells'])
    lines = ['| measurement | without (lab plan 9e4f793a) | with KDA checkpoints (lab plan 594faefb) | change |',
             '|---|---:|---:|---:|']
    for label, key in (('cold TTFT 8K', '8192'), ('cold TTFT 32K', '32768'), ('cold TTFT 64K', '65536')):
        c = cold['cells'][key]
        lines.append(f"| {label} | {c['previous_base_s']:.3f} s | {c['final_profile_s']:.3f} s | "
                     f"{pct(c['previous_base_s'], c['final_profile_s'])} |")
    lines.append('| replay TTFT 8K / 32K / 64K | ' + ' / '.join(f"{rep['cells'][k]['previous_base_s']:.3f}" for k in ns)
                 + ' s | ' + ' / '.join(f"{rep['cells'][k]['final_profile_s']:.3f}" for k in ns) + ' s | '
                 + ' / '.join(pct(rep['cells'][k]['previous_base_s'], rep['cells'][k]['final_profile_s']) for k in ns) + ' |')
    lines.append(f'| decode step time, {n} fixed (streams, K) cells | | | median {med:+.2f} %, range {lo:+.2f} to '
                 f'{hi:+.2f} % |')
    lines.append(f"| speculative acceptance (window-wide) | {acc['previous_base']['rate']:.4f} | "
                 f"{acc['final_profile']['rate']:.4f} | {acc['final_profile']['rate'] - acc['previous_base']['rate']:+.4f} |")
    outside = {}
    for who in ('previous_base', 'final_profile'):
        found = []
        for depth, cell in sorted(v['cached_prefix_check'][who]['cells'].items(), key=lambda x: int(x[0])):
            for comp, r in sorted(cell['comparisons'].items()):
                if not r['equal'] and not r['near_tie']:
                    found.append((int(depth), comp, r['first_difference_token'], r['reference_top2_logprob_gap']))
        outside[who] = found

    def describe(found):
        return '; '.join(f"{d // 1024}K {comp.split('_vs_')[0]} against {comp.split('_vs_')[1]}, token {t}, gap {g}"
                         for d, comp, t, g in found) or 'none'
    verdicts = kda['verdicts']
    first = verdicts['rule_fixed_1232']
    failed = [c for c, ok in first['checks'].items() if not ok]
    text = (f"A cached-prefix check compared replayed and extended prompts with fresh ones token by token (a difference "
            f"whose top-2 logprob gap in the reference is below 0.5 counts as a near-tie). Differences outside near-ties: "
            f"with KDA checkpoints {len(outside['final_profile'])} ({describe(outside['final_profile'])}); without "
            f"{len(outside['previous_base'])} ({describe(outside['previous_base'])}). The cause is not established. Under "
            f"the lab's rule as fixed before the window, the KDA-checkpoint plan did not pass clause "
            f"{failed[0].split(' ')[0]} ({failed[0].split(' ', 1)[1]}); it passed a revised clause (no more mismatches than "
            f"the fresh reference, no gap above 2.0) that the lab fixed before seeing that plan's cache-check data. Record: "
            f"`{EVIDENCE['kda']}`.")
    return '\n'.join(lines) + '\n\n' + wrap(text)


def drift_caveat(drift):
    v = drift['values']
    rep = next(r for r in v['same_plan_repeats'] if r.get('hours_apart', 0) >= 2)
    cells = [c['change_pct'] for c in rep['decode_step_time']['cells']]
    shift = v['inferred_window_shift']['judged_drift']['decode_step_time_summary']
    return (f"Room temperature was not logged. Two windows of the same TP4 plan measured {rep['hours_apart']:.1f} hours "
            f"apart differed by {statistics.median(cells):+.2f} % in median decode step time at fixed (streams, K) and "
            f"by {min(cells):+.2f} to {max(cells):+.2f} % in single cells. Windows whose code changes the lab judged not "
            f"to slow decode moved by up to {shift['median']:+.2f} % in median decode step time when measured "
            f"{v['inferred_window_shift']['judged_drift']['hours_apart']:.1f} hours apart (single cells up to "
            f"{shift['max']:+.2f} %), which the lab attributes to the room temperature. The rows below come from "
            f"different days and times of day. Record: `{EVIDENCE['drift']}`.")


def tp2_section(tp2):
    """The rule the TP2 profile was judged by, clause by clause, and what the owner decided."""
    lines = ['| clause | what it checks | measured | result |', '|---|---|---|---|']
    for c in tp2['rule']['clauses']:
        m = c['measured']
        if c['clause'] == '1':
            got = (f"{m['cells']} cells: median {m['median_pct']:+.2f} %, worst cell {m['max_cell']} "
                   f"{m['max_cell_pct']:+.2f} % ({m['max_cell_steps'][0]} / {m['max_cell_steps'][1]} steps)")
        elif c['clause'] in ("2'", '3'):
            got = ' / '.join(f"{m['change_pct'][k]:+.2f} %" for k in ('8192', '32768', '65536')) + ' (8K / 32K / 64K)'
        elif c['clause'] == "4'":
            got = '; '.join(f"{rate} req/s: {m[rate]['candidate']:.2f} against {m[rate]['reference']:.2f}"
                            for rate in sorted(m, key=float))
        elif c['clause'] == '5':
            worst = max(m['cells_change_pct'].items(), key=lambda kv: kv[1])
            got = f"geometric mean {m['geo_pct']:+.2f} %; slowest cell {worst[0]} {worst[1]:+.2f} %"
        elif c['clause'] == '6':
            host = min(m['candidate'], key=lambda h: m['candidate'][h]['min_mib'])
            got = (f"lowest MemAvailable {m['candidate'][host]['min_mib']} MiB on the {host} (the first sample of "
                   f"the run; reference {m['reference'][host]['min_mib']} MiB)")
        elif c['clause'] == "7'":
            outside = [d for d in m['candidate_differences'] if d['classification'] == 'difference outside near-ties']
            got = '; '.join(f"{d['depth'] // 1024}K {d['comparison'].replace('_vs_', ' against ')}: token "
                            f"{d['first_difference_token']}, gap {d['gap']}" for d in outside)
            got = f"{len(outside)} differences outside near-ties ({got})"
        else:
            got = json.dumps(m, sort_keys=True)
        lines.append(f"| {c['clause']} | {c['name']}: {c['rule']} | {got} | {'passed' if c['passed'] else '**failed**'} |")
    return '\n'.join(lines)


def manifest_table(arms, rows):
    """One line per measured arm from the per-arm records (evidence/comparison-arms.json)."""
    by_id = {a.get('id'): a for a in arms['rows']}
    out = ['| row | repository and commit | checkpoint | quantisation | speculative decoding | runtime | TP | context | '
           'admits | KV cache per rank |', '|---|---|---|---|---|---|---|---|---|---|']

    def val(fields, key):
        f = fields.get(key) or {}
        return f.get('value') if f.get('status') == 'recorded' else None
    for row in rows:
        a = by_id[row['id']]
        fl = a['fields']
        ck = fl['model_checkpoint']
        if ck['status'] == 'recorded':
            v = ck['value']
            hf = lambda r: f"[{r}](https://huggingface.co/{r})"
            checkpoint = (f"{hf(v['repository'])} `{v['revision'][:8]}`" if 'repository' in v else
                          f"derived from {hf(v['derived_from']['repository'])} `{v['derived_from']['revision'][:8]}`")
        elif ck['status'] == 'recorded_withheld':
            checkpoint = f"revision `{ck['value']['revision'][:8]}` (repository not named here)"
        else:
            checkpoint = 'not recorded'
        q = val(fl, 'quantisation') or {}
        sd = val(fl, 'speculative_decoding') or {}
        drafter = sd.get('drafter') or {}
        spec = (f"{sd.get('method')}, {drafter.get('repository', '?')} `{str(drafter.get('revision', ''))[:8]}`, "
                f"K {sd.get('num_speculative_tokens')}") if sd else 'not recorded'
        adm = val(fl, 'admission_limit') or {}
        kv = val(fl, 'kv_cache') or {}
        kv_text = f"{kv['gib_per_rank']} GiB" if kv.get('gib_per_rank') else (kv.get('pool') or 'not recorded')
        commit = val(fl, 'measured_commit')
        commit = commit.get('lab_plan') if isinstance(commit, dict) else commit
        out.append(f"| {row['id']} | {val(fl, 'repository')} `{str(commit)[:8]}` | {checkpoint} | {q.get('weights', '-')}"
                   f"{', KV ' + q['kv_cache_dtype'] if q.get('kv_cache_dtype') else ''} | {spec} | {val(fl, 'runtime')} | "
                   f"{val(fl, 'tensor_parallel_size')} | {val(fl, 'context_length')} | "
                   f"{', '.join(f'{k} {v}' for k, v in adm.items())} | {kv_text} |")
    return '\n'.join(out)


def correctness_section(rel):
    """The release window's correctness checks (evidence/release-window-tp4.json) as a table."""
    ko, em = rel['correctness']['korean'], rel['correctness'].get('emoji') or {}
    glitches = ko['isolated_glitches']
    result = f"{ko['runs'] - len(ko['breaks'])} of {ko['runs']} runs complete without a long-generation break"
    if ko['breaks']:
        result += ' (breaks: runs ' + ', '.join(str(b) for b in ko['breaks']) + ')'
    if glitches:
        result += (f"; {len(glitches)} of them with isolated flagged characters ("
                   + ', '.join(f'run {r}: {n}' for r, n in glitches) + ')')
    else:
        result += '; no isolated flagged character'
    lines = ['| check | what ran | result |', '|---|---|---|',
             f"| Korean long generation | " + (f"the Korean prompt 1 at width 6: {ko['runs']} of {ko['expected']} runs, "
                                                f"stopped early" if ko.get('stopped_early') else
                                                f"the Korean prompt 1 x {ko['expected']} at width 6") + f" | {result} |"]
    for cond, c in (em.get('conditions') or {}).items():
        lines.append(f"| emoji / CJK wall, {cond} | {c['n']} requests, {c['tokens']:,} tokens, thinking off | "
                     f"{c['fffd_total']} U+FFFD, {c['malformed']} flagged malformed, {c['errors']} errors, "
                     f"{c['incomplete']} incomplete |")
    return '\n'.join(lines)


def two_windows(rel):
    """Whether the release results come from several boots of the same plan (the record's boots)."""
    return len((rel or {}).get('boots') or []) > 1


def cards(run, runs):
    """'RigMark's JSON receipts and cards (N runs)': a card RigMark did not render is named (the run folder's manifest
    lists it under not_in_window)."""
    try:
        missing = json.loads((HERE / run / 'manifest.json').read_text()).get('not_in_window') or []
    except (OSError, ValueError):
        missing = []
    names = [m['path'].split('/')[-1].replace('.card.txt', '') for m in missing if m['path'].endswith('.card.txt')]
    return (f"RigMark's JSON receipts and cards ({runs} runs" + ''.join(
        f"; RigMark did not render the `{n}` card, the folder's `README.md` says why" for n in names) + ")")


def records_rows(rel, tp4):
    if not rel:
        return f"| `{RUN_TP4}/` | the final TP4 run: RigMark's JSON receipts and cards (2 runs), and llama-benchy's results |"
    return '\n'.join([
        f"| `{RUN_TP4}/` | the release window{'s' if two_windows(rel) else ''}: {cards(RUN_TP4, tp4['runs'])}" + ("; llama-benchy did not run on this plan |" if 'llama_benchy' in (rel.get('not_run') or [])
                                    else ", and llama-benchy's results |"),
        f"| `{RUN_PREVIOUS}/` | the final TP4 run before the release (lab plan `387b3ff3`): RigMark's JSON receipts and cards "
        "(2 runs), and llama-benchy's results |"]
        + ([f"| `{RUN_NVIDIA}/` | the release plan on the NVIDIA checkpoint (lab plan `{_NV['lab_plan']}`): "
            f"{cards(RUN_NVIDIA, _NV['runs'])}" + (", and llama-benchy's results (shown on this page)"
                                                          if LB_NVIDIA else "") + " |"] if RUN_NVIDIA else [])
        + [f"| `{RELEASE}` | the release window{'s' if two_windows(rel) else ''}: plan, context, times, the switch, "
           "served-byte and marker receipts, and the correctness checks |"])


def correctness_text(rel):
    if not rel:
        return ''
    order, ko = rel.get('order') or [], rel['correctness']['korean']
    before = fast(rel) and 'korean' in order and 'rigmark' in order and order.index('korean') < order.index('rigmark')
    where = (f"In a separate boot of the same plan (the correctness window, {rel['dates'].get('correctness_window')}):"
             if two_windows(rel) else "In the release window's boot, before the speed runs:" if before else
             "In the release window's boot, after the speed runs:")
    emoji = 'emoji' in rel['correctness']
    intro = wrap(f"{where} a Korean long-generation prompt (sampled with "
                 f"the server's defaults, the test that showed long-generation breaks in earlier windows)"
                 + (" and the emoji and CJK wall stress test (thinking off)" if emoji else "") + ". "
                 + (f"The Korean check was stopped early by the owner's choice after {ko['runs']} of its {ko['expected']} "
                    f"planned runs; the other {ko['expected'] - ko['runs']} were not run. " if ko.get('stopped_early') else "")
                 + ("The emoji and CJK wall stress test was not run on this plan. "
                    if 'emoji' in (rel.get('not_run') or []) else "")
                 + f"A break follows the lab's rule: a derailed run, a runaway, a "
                 f"failed request, or a "
                 f"garbled run with at least 5 flagged characters (U+FFFD or Cyrillic, Arabic, Thai or kana characters); "
                 f"fewer flagged characters are an isolated glitch. Record: `{RELEASE}`.")
    return '\n## Correctness checks (TP4 release window)\n\n' + intro + '\n\n' + correctness_section(rel) + '\n'


def last_day(rel):
    """The last measurement day: 2026-09-25, or the day a fast release window ended when later."""
    days = ['2026-09-25']
    if fast(rel):
        days += re.findall(r'\d{4}-\d\d-\d\d', ' '.join([str(rel['dates'].get('window'))]
                                                        + [str(b.get('when')) for b in rel.get('boots') or []]))
    return max(days)


def fast_row_text(rel, tp4):
    """The TP4 row's bullet for the fast release set: what ran in the release window, in order, and what did not."""
    d, ko = rel['dates'], rel['correctness']['korean']
    order = [k for k in rel.get('order') or [] if k in ORDER_TEXT]
    part = {'korean': 'the correctness checks below' + (f" (the Korean check stopped early after {ko['runs']} of "
                                                         f"{ko['expected']} runs)" if ko.get('stopped_early') else ''),
            'emoji': 'the emoji and CJK wall stress test',
            'rigmark': f"{tp4['runs']} runs of every RigMark cell ({d.get('rigmark') or 'times in the record'}; the median "
                       "is shown)",
            'llama_benchy': 'llama-benchy',
            'sparkdash': (f"sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol, "
                          f"{rel['benchmarks']['sparkdash_rounds']} rounds; each cell is the median of its rounds)"),
            'tool_eval': 'tool-eval-bench'}
    if 'korean' in order and 'emoji' in order:   # both are the correctness checks below
        order.remove('emoji')
    window = d.get('window') or ''
    when = f"one boot from {window}" if ' to ' in window else f"one boot on {d['day']}"
    nv = rel.get('nvidia_arm') or {}
    return (f"- **The RiNGSiDE TP4 row** is the release window of the TP4 profile's own plan, lab plan "
            f"`{rel['lab_plan']}`: {when} with, in this order, {and_list(part[k] for k in order)}. "
            + (not_run_sentence(rel) + ("; the llama-benchy results below are the release plan's run on the NVIDIA "
                                        f"checkpoint (lab plan `{nv.get('lab_plan')}`)" if LB_NVIDIA else "") + ". "
               if rel.get('not_run') else "")
            + "The replay matrix was not run on it. The final TP4 run of lab plan `387b3ff3` (the profile before the "
            "release's changes) is listed with the additional arms" + (
                f", and so is the release plan's RigMark run on the NVIDIA checkpoint (lab plan `{_NV['lab_plan']}`)."
                if RUN_NVIDIA else "."))


def release_row_text(rel, tp4):
    """The bullet that describes the RiNGSiDE TP4 row when it is the release window."""
    if fast(rel) and not two_windows(rel):
        return fast_row_text(rel, tp4)
    d = rel['dates']
    two = two_windows(rel)
    return (f"- **The RiNGSiDE TP4 row** is the release window{'s' if two else ''} of the TP4 profile's own plan, "
            f"lab plan `{rel['lab_plan']}`: one boot on {d['day']} with {tp4['runs']} runs of every RigMark cell "
            f"({d.get('rigmark') or 'times in the record'}; the median is shown), then llama-benchy, tool-eval-bench"
            f"{',' if not two else ' and'} sparkDash (a Python reproduction "
            f"of MiaAI-Lab/sparkDash's decode-benchmark protocol, {rel['benchmarks']['sparkdash_rounds']} rounds; each "
            f"cell is the median of its rounds)" + (
                f". The results come from {len(rel['boots'])} boots of the same plan, each from the window that "
                f"completed it: {rel['boots_text']}." if two else " and the correctness checks below, in that order.")
            + " The replay matrix was not run on it. The final TP4 run of lab plan `387b3ff3` (the profile before the "
            "release's changes) is listed with the additional arms" + (
                f", and so is the release plan's RigMark run on the NVIDIA checkpoint (lab plan `{_NV['lab_plan']}`)."
                if RUN_NVIDIA else "."))


def tool_eval_where(rel):
    """Where the release window's tool-eval-bench run ran: after llama-benchy, or (fast release set) after the phases
    the record's order puts before it."""
    order = rel.get('order') or []
    if not fast(rel) or 'tool_eval' not in order:
        return "the release window's boot after llama-benchy, "
    before = [ORDER_TEXT[k] for k in order[:order.index('tool_eval')] if k in ORDER_TEXT]
    return "the release window's boot" + (f" after {and_list(before)}" if before else "") + ", "


def llama_benchy_nvidia_text(rel):
    """The llama-benchy paragraph when llama-benchy did not run on the plan: the release plan's run on the NVIDIA
    checkpoint (the record's nvidia_arm)."""
    nv = rel['nvidia_arm']
    ck = nv.get('checkpoint') or {}
    return (f"[eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0 did not run on the TP4 profile's plan "
            f"(lab plan `{rel['lab_plan']}`). The results below are the release plan's run on the NVIDIA checkpoint: lab "
            f"plan `{nv['lab_plan']}` with `{ck.get('repository')}` revision `{str(ck.get('revision'))[:8]}` (the TP4 "
            f"profile is that plan with the checkpoint switch), window {nv['window']} "
            f"({nv.get('llama_benchy_when') or nv.get('when')}, UTC+7), in the same boot as that arm's RigMark cells, on "
            f"llama-benchy's own prompts (English prose from its default book) with thinking on. Its numbers measure "
            f"different prompts, request shapes and statistics than RigMark's and do not compare cell for cell with the "
            f"tables above; no comparison recipe has been run with it here. Medians of 3 runs; arguments and files: "
            f"`{RUN_LLAMA_BENCHY}/README.md` (the final TP4 run's are in `{RUN_PREVIOUS}/`, the previous TP4 profile's "
            f"llama-benchy results are in `{RUN5}/`).")


def profile_note(plan):
    """What the TP4 profile adds to the plan of the TP4 row, from its source plan (launch/profiles/tp4/profile.json)."""
    derived = plan.get('derived_from')
    if not derived:
        return ''
    if derived.get('base_derived_from'):  # the RiNGSiDE release plan: the release changes plus served fixes
        steps = derived['steps']
        return (f" The TP4 profile of this repository, lab plan `{plan['sha256'][:8]}`, was not benchmarked: it is the "
                f"RiNGSiDE release plan `{derived['plan_sha256'][:8]}` (lab plan `6f490797` with change notices and the "
                f"DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code) with {len(steps)} "
                f"served changes, each checked on its own "
                f"(`CURRENT.md`).")
    return (f" The TP4 profile of this repository, lab plan `{plan['sha256'][:8]}`, was not benchmarked: it is "
            f"{derived['note']}")


TP2_ROW_PLAN = '69ad74fc'   # the plan of the RiNGSiDE TP2 row (its final run)


def model_line(arms):
    """The checkpoint of the RiNGSiDE TP4 and TP2 rows, from their arms (evidence/comparison-arms.json)."""
    by = {a.get('id'): a for a in arms['rows']}
    ck = []
    for rid in ('tp4-ringside', 'tp2-ringside'):
        v = (((by.get(rid) or {}).get('fields') or {}).get('model_checkpoint') or {}).get('value') or {}
        ck.append((v.get('repository'), (v.get('revision') or '')[:8]))
    if ck[0] == ck[1]:
        return (f"- Model `{ck[0][0]}` revision `{ck[0][1]}`, drafter `incoai/GLM-5.3-Flash-DFlash2` revision\n"
                f"  `7d74cdd8` for the RiNGSiDE profiles.")
    return wrap(f"- Model `{ck[0][0]}` revision `{ck[0][1]}` for the RiNGSiDE TP4 row and `{ck[1][0]}` revision "
                f"`{ck[1][1]}` for the TP2 row, drafter `incoai/GLM-5.3-Flash-DFlash2` revision `7d74cdd8` for both.",
                '  ')


def profile_note_tp2(plan, withheld=False):
    """What the TP2 profile adds to the plan of the TP2 row, from its source plan (launch/profiles/tp2/profile.json);
    a profile whose status is `withheld` says so."""
    derived = plan.get('derived_from')
    if not derived or plan['sha256'].startswith(TP2_ROW_PLAN):
        return (' **The TP2 profile of this repository is withheld from this release** (`README.md`).' if withheld
                else '')
    steps = derived.get('steps') or []
    fixes = [s for s in steps if s.get('component')] if any(s.get('component') for s in steps) else steps
    served_on = ''
    ck = next((s for s in steps if s.get('checkpoint')), None)
    if ck is not None:   # a TP2 profile on another checkpoint: the checkpoint and its argument change, not measured
        import re
        m = re.match(r'the checkpoint (\S+) \(revision (\w+)', ck['summary'])
        args = [a for s in steps if s.get('arguments') and not s.get('component') for a in s['arguments']]
        served_on = (f", served on the checkpoint `{m.group(1)}` revision `{m.group(2)}`"
                     + (' with ' if args else '') + ' and '.join(
                         f"`{a['flag']} {a['to']}`" + (f" (was `{a['from']}`)" if a['from'] is not None else '')
                         for a in args)
                     + "; the TP2 numbers of this profile follow")
    what = (f" the TP2 release plan `{derived['plan_sha256'][:8]}` (lab plan `{TP2_ROW_PLAN}` with change notices and "
            f"the DFlash2 capture and drafter KV-cache group written from vLLM's DeepSeek-V4 code) with {len(fixes)} "
            f"correctness fixes, each verified by GPU leaves and CPU "
            f"checks (`CURRENT.md`){served_on}." if derived.get('base_derived_from') else ' ' + derived['note'])
    if withheld:
        return (f" **The TP2 profile of this repository, lab plan `{plan['sha256'][:8]}`, is withheld from this release** "
                f"(`README.md`); it was not benchmarked and is" + what)
    return f" The TP2 profile of this repository, lab plan `{plan['sha256'][:8]}`, was not benchmarked: it is" + what


def render(root=HERE):
    data, ev = load(root)
    rows = data['rows']
    group = {'tp4': [r for r in rows if r['tp'] == 'TP4' and r['table'] == 'main'],
             'tp2': [r for r in rows if r['tp'] == 'TP2' and r['table'] == 'main'],
             'extra': [r for r in rows if r['table'] == 'extra']}
    tp4 = next(r for r in rows if r['id'] == 'tp4-ringside')
    tp2 = next(r for r in rows if r['id'] == 'tp2-ringside')
    rel = ev['release']
    te = ev['tool_eval']
    profile4 = json.loads((root.parents[1] / 'launch/profiles/tp4/profile.json').read_text())['source_plan']
    prof2 = json.loads((root.parents[1] / 'launch/profiles/tp2/profile.json').read_text())
    profile2, withheld2 = prof2['source_plan'], prof2.get('status') == 'withheld'
    tp2_is_profile = profile2['sha256'].startswith(TP2_ROW_PLAN)
    many, caps = many_streams(data)
    promo = ev['tp2']
    info = promo['headline_changes']['information_not_gated']
    c6 = info['short_code_aggregate_tok_s']['6']
    l6 = promo['headline_changes']['staggered']['cells_change_pct']['L6 prefill-first newcomer']
    rounds = ev['tp2_previous_row']['values']['staggered_rounds']['6']['prefill_first_newcomer_ttft_s']
    prev2 = next(r for r in rows if r['id'] == 'tp2-previous-profile')
    swept = prev2['replay_matrix'].get('swept') or []
    final2 = ev['tp2_row']
    return f"""# Results

{wrap(f"Measured on four (TP4) and two (TP2) NVIDIA DGX Spark systems (GB10, 128 GB unified memory each) cabled as a "
       f"switchless ring, 2026-09-21 to {last_day(rel)}. " + (
       f"**The RiNGSiDE TP4 row's RigMark cells are medians of {tp4['runs']} runs** (the release window of the TP4 "
       f"profile's plan, one boot; receipts in `{RUN_TP4}/`), **the RiNGSiDE TP2 row's of {tp2['runs']} runs** (its final "
       f"run, one boot), the final TP4 run before the release (lab plan `387b3ff3`, an additional arm) holds medians of "
       f"2 runs (receipts in `{RUN_PREVIOUS}/`), " + (
           f"the release plan's run on the NVIDIA checkpoint (lab plan `{_NV['lab_plan']}`, an additional arm) holds "
           f"medians of {_NV['runs']} runs (receipts in `{RUN_NVIDIA}/`), " if RUN_NVIDIA and _NV_REC else "")
       + f"the previous profiles' rows " if rel else
       f"**The RiNGSiDE TP4 and TP2 rows' RigMark cells are medians of {tp4['runs']} "
       f"runs** (the final run of each profile, one boot each; TP4 receipts in `{RUN_TP4}/`), the previous profiles' rows ") +
       f"(additional arms) hold medians of 5 runs (TP4, receipts in `{RUN5}/`) and of 3 staggered rounds (TP2), and "
       f"**every other cell is a single run**; the runs column of each table says which. `results.json` holds every "
       f"number with the lab window it comes from, `evidence/` the records behind the derived figures, the protocol of "
       f"the replay matrix, the tool-eval-bench runs and a manifest of every measured arm, and `runs/` the RigMark "
       + ("and " if not LB_NVIDIA else "receipts ")
       + (f"llama-benchy receipts of the {'three' if rel else 'two'} multi-run TP4 windows." if not LB_NVIDIA else
          f"of the {WORDS[len(RUNS)]} multi-run TP4 windows, and the llama-benchy receipts of {WORDS[len(RUNS) - 1]} of "
          f"them (not the "
          f"release window's: llama-benchy did not run on this plan)."))}

> **NON-DETERMINISTIC outputs.** Both profiles accumulate NVFP4 MoE outputs with atomic adds in prefill (eager
> prefill launches and the prefill rows of mixed prefill and decode steps), and the TP4 profile also in decode launches
> of 10 or more rows (`B12X_GLM53_ATOMIC_DECODE_MIN_ROWS=10`), so the order varies from run to run and outputs are not
> bit-reproducible. TP4 decode launches below 10 rows and all TP2 decode launches stay deterministic.

## What is compared

Each row is a **whole recipe** as it ran on this hardware: its own model checkpoint, quantisation, speculative
decoding, image, runtime, parallel layout and admission limit. A difference between two rows is the difference
between two recipes, not between two kernels or two settings. The comparison recipes are independent projects with
other design goals (EXL3 quantization, 1M-token KV); they are named by GitHub owner and repository with the commit
measured, and the configuration column and the manifests below say how each was set up. They admit fewer concurrent
requests than the RiNGSiDE TP4 profile, so the tables stop at 6 streams.

## How it was measured

- **RigMark** ([alexellis/rigmark](https://github.com/alexellis/rigmark)), in the fork
  [{FORK['repository']}](https://github.com/{FORK['repository']}) (branch `{FORK['branch']}`, commit
  [`{FORK['commit'][:8]}`](https://github.com/{FORK['repository']}/commit/{FORK['commit']}), parent
  alexellis/rigmark `{FORK['parent'][:8]}`). The windows record the local revision `{FORK['measured'][:8]}`; `{FORK['commit'][:8]}` was
  published with the identical tree (`{FORK['tree'][:8]}`). Protocol 1.1.0, prompts 1.0.0, `reasoning_effort` low in
  the chat template arguments. Cells: cold time to first token at 8K, 32K and 64K prompt tokens; single-stream
  decode rate on code, prose and structured prompts; aggregate end-to-end throughput of short code requests at 1 to 6
  concurrent streams (8, 12 and 16 for the RiNGSiDE TP4 profile).
- **Staggered arrivals** (the fork's section). *Prefill-first*: one 32,768-token request starts, and while it is in its
  prefill L-1 short requests arrive; the cell is the median time to first token of those short newcomers.
  *Decode-first*: L-1 short streams are decoding when a 32,768-token request arrives; the cell is that request's time
  to first token. L = 2, 4, 6 (also 8, 12, 16 for the RiNGSiDE TP4 profile). "-" means no valid cell (the round's
  newcomer only started after every incumbent had finished).
- **sparkDash**: the lab's reproduction, in Python, of the decode benchmark of MiaAI-Lab/sparkDash (its DecodeBench
  protocol: the same prompts, temperature 0, 400 tokens, thinking off, one warm-up stream); code prompts, one stream.
- **Replay matrix** (RiNGSiDE profiles only): a replay of authored multi-user scenarios with a 16-task quality set and
  an arrival-rate sweep against a per-request latency target. Protocol: `../protocols/README.md`; record:
  `{EVIDENCE['replay']}`.
{model_line(ev["arms"])}
{wrap(release_row_text(rel, tp4) if rel else (f"- **The RiNGSiDE TP4 row** is the final TP4 run: lab plan `387b3ff3`, one boot on 2026-09-25 with {tp4['runs']} runs of "
       f"every RigMark cell (00:30 to 00:59; the median is shown), then llama-benchy and tool-eval-bench in the same boot. "
       f"Lab plan `6f490797` is this plan with the replay-boundary admit fix: `glm53_replay_boundary.py` "
       f"`d41662b1` instead of `144c7d34`, every other served file and every flag equal. In the final run the module "
       f"refused five valid prefix hits at the 1,152-token boundary during tool-eval-bench (none during RigMark), and each "
       f"of those requests was recomputed; the fix accepts the image's own cache entry under the same hash. The fix was "
       f"not measured with RigMark: the lab judged that its code path only acts when an image cache entry and a replay "
       f"boundary share a hash, which RigMark's prompts do not produce. Its check window ran the cache check and "
       f"tool-eval-bench (below), with no refused hit. "
       + ("sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) ran once on lab plan "
          "`6f490797` in a separate window on 2026-09-25, and its cell is marked with that "
          "plan; the replay matrix was not run on either plan." if tp4.get('sparkdash_window')
          else "sparkDash and the replay matrix were not run on the final profile.")
       + profile_note(profile4)), '  ')}
- **The previous TP4 profile** (lab plan `50e80d44`, 2026-09-24) is listed with the additional arms: RigMark and the
  staggered arrivals from one boot with 5 runs of every cell, and sparkDash and the replay matrix from the final run of
  lab plan `9e4f793a` (the same profile without the KDA checkpoints), single runs, marked with that plan. A single-run
  window of the same code is listed there too: lab plan `594faefb`, whose copy of `mhc_prefill_sharding.py`
  differs only in its header comment.
{wrap(f"- **The RiNGSiDE TP2 row** is the final TP2 run: lab plan `{TP2_ROW_PLAN}` "
       + ('(the TP2 profile itself)' if tp2_is_profile else "(the base of the TP2 profile, before the release's "
                                                          'changes and fixes)') + ", one boot on 2026-09-25 "
       f"with {tp2['runs']} runs of every RigMark cell (02:27 to 02:52; prefill at 8K, 32K and 64K). It was one window "
       f"without a fresh reference window (the owner's decision), so no rule judged it; the TP2 section compares it, as "
       f"information only, with the stored windows of the previous TP2 profile. "
       + ("sparkDash (a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol) ran once in a separate "
          "window of the same plan on 2026-09-25; the replay matrix was not run on it." if tp2.get('sparkdash_window')
          else "sparkDash and the replay matrix were not run on it."), '  ')}
{wrap(f"- **The previous TP2 profile** (lab plan `32dd2ab8`, 2026-09-24) is listed with the additional arms: RigMark, "
       f"staggered arrivals over three rounds and the replay matrix swept at {', '.join(f'{x:g}' for x in swept)} req/s "
       f"only, with sparkDash from the final run of the TP2 base before it (lab plan `f1771e0a`), marked with that plan. "
       f"**The owner promoted it although three guard clauses of the lab's rule failed**; the TP2 section lists every "
       f"clause.", '  ')}

## Caveats

{wrap("- **Single runs.** Apart from " + (f"the RiNGSiDE TP4 row (medians of {tp4['runs']} runs), the RiNGSiDE TP2 row and "
       "the final TP4 run before the release (medians of 2 runs)" + (
           f", the release plan's run on the NVIDIA checkpoint (medians of {_NV['runs']} runs)" if RUN_NVIDIA and _NV_REC
           else "") if rel else "the RiNGSiDE TP4 and TP2 rows (medians of "
       "2 runs)") + ", the previous TP4 profile's row (medians of 5 runs) and the previous TP2 profile's staggered cells "
       "(medians of 3 rounds), no cell has repetitions; small differences between single runs are not significant. With "
       "2 runs, the median of a cell is the mean of the two and shows nothing of their spread.", '  ')}
{wrap('- **Room temperature.** ' + drift_caveat(ev['drift']), '  ')}
- **Throughput depends on the prompt text.** Speculative decoding accepts more or fewer drafted tokens depending on
  what is generated, so decode rates and aggregate throughput depend on the prompt text as well as on the recipe;
  the cells hold for the benchmarks' own prompts.
- **Request caps.** The comparison recipes ran with at most 4 (MiaAI-Lab, their `MAX_NUM_SEQS=4`; 2 in their
  long-coding example) or 6 (tonyd2wild) concurrent requests. For tonyd2wild's TP4 recipe the 6 slots at 262,144
  context were the lab's choice (the published launch uses 64 slots at 500,000); the TP2 recipe ships 6. The RiNGSiDE
  TP2 profile also admits 6, its TP4 profile 16. Up to 6 streams every recipe ran every cell (a recipe that admits
  fewer queues the rest); streams beyond a recipe's cap were not run for the comparison recipes.

## Ranking

Rows are ordered by mean rank (best first). For each of the {len(METRICS)} ranked columns (cold TTFT 8K / 32K / 64K;
decode code / prose / structured; short code aggregate C1 and C6; staggered prefill-first and decode-first at L2, L4
and L6; sparkDash code x1) the rows with a value are ranked from 1 (best: the lowest time, the highest throughput);
equal values share the mean of their positions. A row's mean rank is the mean over the columns where it has a value.
A missing cell ("-", "not run") is left out of that row's mean: it neither counts against the row nor for it, which
can favour a row whose missing cell would have ranked low. The last column gives the mean rank and the number of
columns it is taken over. Cells marked with another lab plan count like the others. The admission limit and the replay
matrix are not ranked.

## TP4

> Decode and aggregate throughput depend on the prompt text through speculative acceptance (Caveats).

{table(group['tp4'])}

### The RiNGSiDE TP4 profile at 8 to 16 streams ({'release window' if rel else 'final TP4 run'}, medians of {tp4['runs']} runs)

{many}

{wrap('Not run for the comparison recipes: ' + caps + '.')}

## TP2

> Decode and aggregate throughput depend on the prompt text through speculative acceptance (Caveats).

{table(group['tp2'])}

{wrap((f"The TP2 profile (lab plan `{TP2_ROW_PLAN}`) is" if tp2_is_profile else
        f"Lab plan `{TP2_ROW_PLAN}`, the base of the TP2 profile, is")
       + f" the previous TP2 profile (`32dd2ab8`) plus the startup warm-up v3 "
       f"(six prompts), the replay boundary v3 with the admit fix (at most 6 states kept) and P1 first-token repay, as "
       f"in the TP4 profile; TP2 has neither the fp16 KDA state nor the greedy argmax verification. **Its final run had "
       f"no fresh reference window** (the owner's decision), so "
       f"no rule judged it. As information only, the lab compared it with the stored windows of the previous profile "
       f"(different boots, times of day and run counts; a ratio per cell, above 0 % = the final run is faster, geometric "
       f"means per group): {final2['information_comparison']['line']}. The replay-boundary cache check of the final run "
       f"found the predicted hits with {final2['values']['cache_check']['mismatches']} differences from fresh prefills "
       f"and {final2['values']['cache_check']['near_ties']} near-tie, and no hit was refused. Record: "
       f"`{EVIDENCE['tp2_row']}`." + profile_note_tp2(profile2, withheld2))}

The previous TP2 profile (lab plan `32dd2ab8`), on which the current one builds, is the TP2 base before it
(`f1771e0a`) plus prefix-cache retention every 4,608 tokens, the prefix-cache contract and the MoE mixed-batch split.
It was measured against a fresh window of that base on the same node pair, back to back, and judged by the lab's rule,
fixed before the data and amended before any serving data. **Three clauses failed, and the owner promoted it
anyway:**

{tp2_section(promo)}

{wrap(f"Not gated by the rule, and slower against the reference: the short-code aggregate at 6 streams moved "
        f"{c6['change_pct']:+.2f} % ({c6['reference']:.1f} to {c6['candidate']:.1f} tok/s, single runs), and the L6 "
        f"prefill-first newcomer {l6:+.2f} % (median of three rounds: {' / '.join(f'{x:.2f}' for x in rounds)} s). Still "
        f"open: whether the cached 4,608-token state gives the same next tokens as a fresh prefill, and the memory "
        f"headroom of the rank 0 host. Record: `{EVIDENCE['tp2']}`.")}

### Prefill by prompt length ({'release window' if rel else 'final TP4 run'}, medians of {tp4['runs']} runs)

{wrap("RigMark's prefill suite in the same boot: time to first token of a prompt new to the cache (cold) and of the same "
       "prompt sent again right away (warm replay, served from the prefix cache). The largest prompt, 262,136 tokens, is 8 "
       "tokens under the 262,144-token context" + (f" of the earlier profiles; the profile serves {rel['context']:,} tokens, "
       "which the prefill suite does not reach." if rel and rel['context'] != 262144 else "."))}

{prefill_table(tp4)}

### llama-benchy ({"the release plan on the NVIDIA checkpoint" if LB_NVIDIA else
                    ('release window' if rel else 'final TP4 run') + ", same boot"})

{wrap(llama_benchy_nvidia_text(rel) if LB_NVIDIA else f"[eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.4.0 in the same boot as the "
       f"{'release window' if rel else 'final run'}'s RigMark "
       f"cells, on its own prompts (English prose from its default book) with thinking on. Its numbers measure different "
       f"prompts, request shapes and statistics than RigMark's and do not compare cell for cell with the tables above; no "
       f"comparison recipe has been run with it here. Medians of 3 runs; arguments and files: `{RUN_TP4}/README.md` (the "
       + (f"final TP4 run's are in `{RUN_PREVIOUS}/`, the " if rel else "") +
       f"previous TP4 profile's llama-benchy results are in `{RUN5}/`).")}

{llama_benchy_table(ev['llama_benchy'])}

### tool-eval-bench (TP4)

{wrap(f"[{te['tool']['repository']}](https://github.com/{te['tool']['repository']}) commit `{te['tool']['commit'][:8]}` "
       f"({te['tool']['licence']}): {te['tool']['scenarios']} tool-calling scenarios (tool selection, parameter "
       f"precision, multi-step chains, refusal, error recovery, instruction following, prompt-injection safety, "
       f"structured output, planning), pass 2 / partial 1 / fail 0 points, at temperature 0 with seed 42 and thinking on "
       f"(both recipes think by default), 4 scenarios in parallel. One run per window: "
       + (tool_eval_where(rel) if rel else "") + f"the final TP4 run's boot after "
       f"llama-benchy, the TP4 profile's own check window, and the MiaAI-Lab TP4 recipe with the launch plan of its TP4 "
       f"row; no tonyd2wild recipe was run with it. "
       + ('The same three scenarios failed in every run. ' if te['same_failed_scenarios_in_every_run'] else '')
       + f"The median turn time depends on each recipe's speed and on how long its model thinks. Record: "
       f"`{EVIDENCE['tool_eval']}`.")}

{tool_eval_table(te)}
{correctness_text(rel)}
## Additional measured arms

{wrap("Not the headline rows: " + ("the final TP4 run before the release (lab plan `387b3ff3`, 2 runs), " if rel else "")
       + (f"the release plan's run on the NVIDIA checkpoint (lab plan `{_NV['lab_plan']}`, {_NV['runs']} runs), "
          if rel and RUN_NVIDIA and _NV_REC else "")
       + "the previous TP4 profile (5 runs) and a single-run window of its code, the earlier TP4 base on its own and with "
       "the optional [msuiche/weightless](https://github.com/msuiche/weightless) GLP-44 steering, the previous TP2 "
       "profile (staggered cells over three rounds), the earlier TP2 base, and a second MiaAI-Lab configuration (their "
       "long-coding example, measured on the other node pair). All single runs except the "
       + (("NVIDIA-checkpoint run's, the " if RUN_NVIDIA and _NV_REC else "")
          + "final TP4 run's and the previous profiles' repeated cells." if rel else "previous profiles' repeated cells."))}

{table(group['extra'])}

## Arm manifests

What each row ran, from `{EVIDENCE['arms']}` (every field there names its source; fields the lab did
not record are marked "not recorded").

{manifest_table(ev['arms'], rows)}

## KDA checkpoints against the TP4 base before them

Measured on 2026-09-24 between 12:48 and 13:30: the TP4 base without them (lab plan `9e4f793a`) and then the
KDA-checkpoint plan (lab plan `594faefb`, the code of the previous TP4 profile `50e80d44`), back to back in one session
(single samples). The final TP4 profile keeps the KDA checkpoints.

{kda_section(ev['kda'])}

## Records

| file | what it holds |
|---|---|
| `results.json` | every number of the tables, with the lab window (`lab_receipt`), lab plan and number of runs of each row |
{records_rows(rel, tp4)}
| `{RUN5}/` | the previous TP4 profile's 5-run window: RigMark's JSON receipts and cards, and llama-benchy's results |
| `{EVIDENCE['tool_eval']}` | tool-eval-bench: per run the score, scenario outcomes, median turn time and settings |
| `{EVIDENCE['arms']}` | per row: repository and commit, checkpoint, quantisation, speculative decoding, image, runtime, parallel layout, context, admission limit, KV cache and every recorded deviation, each with its lab source |
| `{EVIDENCE['kda']}` | the KDA-checkpoint comparison: per-cell decode step times, TTFT, acceptance, the cached-prefix check and the rule verdicts |
| `{EVIDENCE['tp2']}` | the TP2 promotion: every clause of the rule with its threshold, measured value and result |
| `{EVIDENCE['tp2_row']}` | the cells of the RiNGSiDE TP2 row (2 runs) and the information comparison with the previous profile's stored windows |
| `{EVIDENCE['tp2_previous_row']}` | the cells of the previous TP2 profile's row, with all three staggered rounds |
| `{EVIDENCE['replay']}` | the replay matrix: fixture, commands, latency target, sweep and per-window results (protocol text: `../protocols/README.md`) |
| `{EVIDENCE['drift']}` | same-plan repeat windows and the window shifts behind the room-temperature caveat |
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true', help='exit 1 if README.md differs from the rendered page')
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    page = render()
    readme = HERE / 'README.md'
    if a.write:
        readme.write_text(page)
        return 0
    if a.check:
        return 0 if readme.read_text() == page else 1
    sys.stdout.write(page)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
