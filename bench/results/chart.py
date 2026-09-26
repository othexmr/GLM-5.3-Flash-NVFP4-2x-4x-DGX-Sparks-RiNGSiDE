# SPDX-License-Identifier: Apache-2.0
"""Render the results chart of the top-level README from this recipe's own records: bench/results/chart-light.svg,
bench/results/chart-dark.svg and the <picture> block between the chart markers in README.md.

    python3 -B bench/results/chart.py --write    # after every data import
    python3 -B bench/results/chart.py --check    # exit 1 if a committed file differs from what the data gives

Inputs (standard library only, no network): results.json (rows tp4-ringside and tp2-ringside, the same source as the
results page), evidence/tool-eval-bench.json (the run of the TP4 row's window), evidence/release-window-tp4.json (the
window's dates) and evidence/comparison-arms.json (the checkpoints). Same data, same bytes.
"""
import argparse
import json
from pathlib import Path
import re
import sys
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
START, END = '<!-- chart:start -->', '<!-- chart:end -->'
SERIES = (('tp4-ringside', 'tp4'), ('tp2-ringside', 'tp2'))   # (results.json row id, colour slot); TP4 is required
OUTPUTS = {'light': 'chart-light.svg', 'dark': 'chart-dark.svg'}
WIDTH = 880
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
# Two themes, each its own steps of the same two hues (validated for both surfaces); ink, grid and axis are neutral.
THEMES = {
    'light': {'surface': '#fcfcfb', 'border': '#e1e0d9', 'ink': '#0b0b0b', 'ink2': '#52514e', 'muted': '#898781',
              'grid': '#e1e0d9', 'axis': '#c3c2b7', 'tp4': '#2a78d6', 'tp2': '#eb6834'},
    'dark': {'surface': '#1a1a19', 'border': '#383835', 'ink': '#ffffff', 'ink2': '#c3c2b7', 'muted': '#898781',
             'grid': '#2c2c2a', 'axis': '#383835', 'tp4': '#3987e5', 'tp2': '#d95926'},
}
WORKLOADS = ('code', 'prose', 'structured', 'json')   # display order; other sparkDash workloads follow, sorted


def load(root=ROOT):
    res = root / 'bench/results'
    rows = {r['id']: r for r in json.loads((res / 'results.json').read_text())}
    ev = res / 'evidence'
    tool = json.loads((ev / 'tool-eval-bench.json').read_text())
    rel = json.loads((ev / 'release-window-tp4.json').read_text()) if (ev / 'release-window-tp4.json').exists() else None
    arms_path = ev / 'comparison-arms.json'
    arms = {r['id']: r for r in json.loads(arms_path.read_text())['rows']} if arms_path.exists() else {}
    return rows, tool, rel, arms


def checkpoint(arms, row_id):
    value = (((arms.get(row_id) or {}).get('fields') or {}).get('model_checkpoint') or {}).get('value') or {}
    return value.get('repository'), (value.get('revision') or '')[:8]


def series(root=ROOT):
    """The rows shown, TP4 first, with their labels and provenance lines."""
    rows, tool, rel, arms = load(root)
    if SERIES[0][0] not in rows:
        raise SystemExit(f'results.json has no row {SERIES[0][0]}: nothing to chart')
    out = []
    for row_id, slot in SERIES:
        row = rows.get(row_id)
        if row is None:
            continue
        repo, rev = checkpoint(arms, row_id)
        owner = repo.split('/')[0] if repo else None
        owner = {'nvidia': 'NVIDIA'}.get(owner, owner)
        plan = f"lab plan {row['lab_plan']}" if row.get('lab_plan') else f"window {row['lab_receipt']}"
        label = row['tp'] + ' (' + (f'{owner} checkpoint, ' if owner else '') + plan + ')'
        where = f"window {row['lab_receipt']}"
        if rel and rel.get('window') == row['lab_receipt']:
            where = f"{rel['dates']['window']} ({rel['dates']['timezone']}), window {rel['window']}"
        runs = f"RigMark cells are medians of {row['runs']} runs" if row.get('runs', 1) > 1 else 'RigMark single run'
        detail = [(row.get('configuration') or row['tp']).split(';')[0],
                  ' \u00b7 '.join(x for x in (f'checkpoint {repo} rev {rev}' if repo else '', runs) if x), where]
        out.append({'id': row_id, 'slot': slot, 'tp': row['tp'], 'label': label, 'detail': detail, 'row': row})
    runs = [r for r in tool['runs'] if r['window'] == rows[SERIES[0][0]]['lab_receipt']]   # every round of that window
    run = sorted(runs, key=lambda r: r.get('round', 1)) if runs else None
    return out, run


# ---------------------------------------------------------------------------------------------------------- drawing

def n(v):
    s = f'{v:.1f}'
    return s[:-2] if s.endswith('.0') else s


def nice_step(maxv, ticks):
    raw = maxv / ticks
    mag = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def scale_max(maxv, ticks=4, head=1.08):
    step = nice_step(maxv, ticks)
    top = step
    while top < maxv * head:
        top += step
    return top, step


def tick_text(v):
    return f'{v:g}'


def depth_label(tokens):
    return f'{round(tokens / 1024)}K'


class Svg:
    def __init__(self, theme):
        self.t = THEMES[theme]
        self.parts = []

    def text(self, x, y, s, size=11.5, fill='ink', weight=None, anchor=None, halo=False):
        attrs = f'x="{n(x)}" y="{n(y)}" font-size="{n(size)}" fill="{self.t[fill]}"'
        if weight:
            attrs += f' font-weight="{weight}"'
        if anchor:
            attrs += f' text-anchor="{anchor}"'
        if halo:   # a surface-coloured outline keeps a value label legible where it crosses a line
            attrs += f' stroke="{self.t["surface"]}" stroke-width="3" stroke-linejoin="round" paint-order="stroke"'
        self.parts.append(f'<text {attrs}>{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke='grid', width=1):
        self.parts.append(f'<line x1="{n(x1)}" y1="{n(y1)}" x2="{n(x2)}" y2="{n(y2)}" stroke="{self.t[stroke]}" '
                          f'stroke-width="{n(width)}"/>')

    def path(self, d, fill, tip, stroke=None):
        extra = f' stroke="{self.t[stroke]}" stroke-width="2"' if stroke else ''
        self.parts.append(f'<path d="{d}" fill="{self.t[fill]}"{extra}><title>{escape(tip)}</title></path>')

    def hbar(self, x0, y, length, h, fill, tip):
        r = min(4, h / 2, length)
        x1 = x0 + length
        d = (f'M{n(x0)} {n(y)}H{n(x1 - r)}A{n(r)} {n(r)} 0 0 1 {n(x1)} {n(y + r)}V{n(y + h - r)}'
             f'A{n(r)} {n(r)} 0 0 1 {n(x1 - r)} {n(y + h)}H{n(x0)}Z')
        self.path(d, fill, tip)

    def vbar(self, x, base, height, w, fill, tip):
        r = min(4, w / 2, height)
        top = base - height
        d = (f'M{n(x)} {n(base)}V{n(top + r)}A{n(r)} {n(r)} 0 0 1 {n(x + r)} {n(top)}H{n(x + w - r)}'
             f'A{n(r)} {n(r)} 0 0 1 {n(x + w)} {n(top + r)}V{n(base)}Z')
        self.path(d, fill, tip)

    def polyline(self, points, stroke):
        pts = ' '.join(f'{n(x)},{n(y)}' for x, y in points)
        self.parts.append(f'<polyline points="{pts}" fill="none" stroke="{self.t[stroke]}" stroke-width="2" '
                          f'stroke-linejoin="round" stroke-linecap="round"/>')

    def dot(self, x, y, fill, tip):
        self.parts.append(f'<circle cx="{n(x)}" cy="{n(y)}" r="4" fill="{self.t[fill]}" stroke="{self.t["surface"]}" '
                          f'stroke-width="2"><title>{escape(tip)}</title></circle>')

    def swatch(self, x, y, fill):
        self.parts.append(f'<rect x="{n(x)}" y="{n(y)}" width="12" height="12" rx="3" fill="{self.t[fill]}"/>')


def panel_title(svg, x, y, title, subtitle=None):
    svg.text(x, y, title, 13.5, weight='600')
    if subtitle:
        svg.text(x, y + 17, subtitle, 11.5, fill='ink2')


def y_axis(svg, x0, x1, base, height, top, step, unit=None):
    v = 0
    while v <= top + 1e-9:
        y = base - height * v / top
        svg.line(x0, y, x1, y, 'axis' if v == 0 else 'grid')
        svg.text(x0 - 6, y + 4, tick_text(v), 11, fill='ink2', anchor='end')
        v += step
    if unit:
        svg.text(x0 - 6, base - height - 10, unit, 11, fill='ink2', anchor='end')


def decode_panel(svg, ss, x, y, w):
    """(a) RigMark single-stream decode, horizontal bars per prompt type."""
    panel_title(svg, x, y, 'Single-stream decode (tok/s)', 'RigMark, one stream, by prompt type')
    cats = [c for c in ('code', 'prose', 'structured') if any(c in s['row']['decode'] for s in ss)]
    top, step = scale_max(max(s['row']['decode'][c] for s in ss for c in cats if c in s['row']['decode']), head=1.0)
    x0, x1 = x + 78, x + w - 44
    bar, gap, group = 15, 2, 14
    y0 = y + 36
    base = y0 + len(cats) * (len(ss) * (bar + gap) - gap) + (len(cats) - 1) * group + 4
    v = 0
    while v <= top + 1e-9:
        gx = x0 + (x1 - x0) * v / top
        svg.line(gx, y0 - 4, gx, base, 'axis' if v == 0 else 'grid')
        svg.text(gx, base + 15, tick_text(v), 11, fill='ink2', anchor='middle')
        v += step
    svg.text(x1, base + 30, 'tok/s', 11, fill='ink2', anchor='end')
    cy = y0
    for c in cats:
        gh = len(ss) * (bar + gap) - gap
        svg.text(x0 - 10, cy + gh / 2 + 4, c, 12, anchor='end')
        for s in ss:
            val = s['row']['decode'].get(c)
            if val is not None:
                length = (x1 - x0) * val / top
                svg.hbar(x0, cy, length, bar, s['slot'], f"{s['tp']} {c}: {val:.1f} tok/s")
                svg.text(x0 + length + 5, cy + bar - 3.5, f'{val:.1f}', 11.5, weight='600', halo=True)
            cy += bar + gap
        cy += group - gap
    return base + 34


def tool_score_text(runs):
    """The score line of the TP4 window's rounds: one value when every round scored the same, else each round's."""
    scores = [r['final_score'] for r in runs]
    points = [f"{r['points']}/{r['max_points']}" for r in runs]
    if len(runs) == 1:
        return f"{scores[0]} ({points[0]} points)"
    if len(set(scores)) == 1 and len(set(points)) == 1:
        return f"{scores[0]} in all {len(runs)} rounds ({points[0]} points each)"
    return ' and '.join(f"{s} ({p} points)" for s, p in zip(scores, points)) + f" in {len(runs)} rounds"


def tool_panel(svg, ss, runs, x, y, w):
    """(e) tool-eval-bench score of the TP4 row's window, every round of it."""
    n = len(runs or [])
    panel_title(svg, x, y, 'tool-eval-bench', f"tool-calling scenarios, {n} run{'s' if n != 1 else ''}")
    if not runs:
        svg.text(x, y + 60, 'no run in the TP4 window', 12, fill='ink2')
        return
    run = runs[0]
    svg.swatch(x, y + 36, ss[0]['slot'])
    svg.text(x + 18, y + 47, ss[0]['tp'] + ', same window as its RigMark cells', 11.5, fill='ink2')
    same = len({(r['final_score'], r['points']) for r in runs}) == 1
    big = str(run['final_score']) if same else ' / '.join(str(r['final_score']) for r in runs)
    svg.text(x, y + 98, big, 46, weight='700')
    off = 64 if same else 28 * len(big) // 2 + 20
    svg.text(x + off, y + 80, 'score' + (f', all {n} rounds' if same and n > 1 else ''), 11.5, fill='ink2')
    svg.text(x + off, y + 98, (f"{run['points']} / {run['max_points']} points" + (' each' if n > 1 else '')) if same
             else ' / '.join(f"{r['points']}" for r in runs) + f" of {run['max_points']} points", 13, weight='600')
    yy = y + 124
    for r in runs:
        c = r['counts']
        label = (f"round {r.get('round', 1)}: " if n > 1 else '')
        svg.text(x, yy, f"{label}{c['passed']} passed, {c['partial']} partial, {c['failed']} failed", 12)
        yy += 18
    failed = sorted({s for r in runs for s in r.get('failed_scenarios') or []})
    if failed:
        svg.text(x, yy, 'failed: ' + ', '.join(failed), 11.5, fill='ink2')
        yy += 18
    started = run.get('started') or ''
    if len(started) >= 16:
        off = started[19:]
        tz = ''
        if re.fullmatch(r'[+-]\d\d:\d\d', off):
            tz = ' (UTC' + off[0] + str(int(off[1:3])) + (':' + off[4:] if off[4:] != '00' else '') + ')'
        svg.text(x, yy, 'started ' + started[:10] + ' ' + started[11:16] + tz, 11.5, fill='ink2')


def prefill_panel(svg, ss, x, y, w):
    """(b) cold time to first token by prompt length: prefill_suite where the row has it, else cold."""
    def cells(row):
        suite = row.get('prefill_suite')
        if suite:
            return {int(k): v['cold_s'] for k, v in suite.items()}
        return {int(k): v for k, v in row['cold'].items()}
    data = {s['id']: cells(s['row']) for s in ss}
    depths = sorted({d for c in data.values() for d in c})
    missing = [s for s in ss if set(depths) - set(data[s['id']])]
    sub = 'RigMark, one request, by prompt length'
    for s in missing:
        have = sorted(data[s['id']])
        sub += f"; {s['tp']}: {depth_label(have[0])}-{depth_label(have[-1])} only"
    odd = [d for d in depths if d % 1024]
    panel_title(svg, x, y, 'Cold prefill: time to first token (s)', sub)
    top, step = scale_max(max(v for c in data.values() for v in c.values()))
    x0, x1 = x + 34, x + w - 6
    height, base = 150, y + 196
    y_axis(svg, x0, x1, base, height, top, step, 's')
    slot = (x1 - x0) / len(depths)
    bw = min(24, (slot - 16) / len(ss))
    for i, d in enumerate(depths):
        cx = x0 + slot * (i + 0.5)
        label = depth_label(d) + ('*' if d in odd else '')
        svg.text(cx, base + 16, label, 11.5, fill='ink2', anchor='middle')
        bx = cx - (bw * len(ss) + 2 * (len(ss) - 1)) / 2
        for s in ss:
            val = data[s['id']].get(d)
            if val is not None:
                h = height * val / top
                svg.vbar(bx, base, h, bw, s['slot'], f"{s['tp']} cold TTFT at {d:,} tokens: {val:.2f} s")
                svg.text(bx + bw / 2, base - h - 5, f'{val:.1f}', 11, weight='600', anchor='middle', halo=True)
            bx += bw + 2
    note = 'prompt tokens'
    if odd:
        note += '; ' + ', '.join(f'{depth_label(d)}* = {d:,}' for d in odd)
    svg.text(x0, base + 32, note, 11, fill='ink2')
    return base + 36


def line_chart(svg, ss, values, x0, x1, base, height, top, step, xs, fmt, tip, ylabels=True, unit=None):
    if ylabels:
        y_axis(svg, x0, x1, base, height, top, step, unit)
    else:
        v = 0
        while v <= top + 1e-9:
            yy = base - height * v / top
            svg.line(x0, yy, x1, yy, 'axis' if v == 0 else 'grid')
            v += step
    lo, hi = xs[0], xs[-1]

    def px(k):
        return x0 + 12 + (x1 - x0 - 24) * (k - lo) / (hi - lo)
    for k in xs:
        svg.text(px(k), base + 20, str(k), 11, fill='ink2', anchor='middle')
    for i, s in enumerate(ss):
        pts = sorted(values[s['id']].items())
        coords = [(px(k), base - height * v / top) for k, v in pts]
        svg.polyline(coords, s['slot'])
        for (k, v), (cx, cy) in zip(pts, coords):
            svg.dot(cx, cy, s['slot'], tip(s, k, v))
            dy = -9 if i == 0 else 16
            svg.text(cx, cy + dy, fmt(v), 11, weight='600', anchor='middle', halo=True)
    return px


def aggregate_panel(svg, ss, x, y, w):
    """(c) aggregate end-to-end throughput of short code requests by concurrency (row field agg)."""
    values = {s['id']: {int(k): v for k, v in s['row']['agg'].items()} for s in ss}
    xs = sorted({k for v in values.values() for k in v})
    panel_title(svg, x, y, 'Aggregate throughput by concurrency (tok/s)',
                'RigMark short code requests, all streams, end to end')
    top, step = scale_max(max(v for c in values.values() for v in c.values()))
    x0, x1 = x + 34, x + w - 6
    height, base = 150, y + 196
    line_chart(svg, ss, values, x0, x1, base, height, top, step, xs, lambda v: f'{v:.0f}',
               lambda s, k, v: f"{s['tp']} at {k} streams: {v:.1f} tok/s", unit='tok/s')
    svg.text(x0, base + 37, 'concurrent streams', 11, fill='ink2')
    return base + 40


def sparkdash_panel(svg, ss, x, y, w):
    """(d) sparkDash mean decode rate per stream, one small chart per workload."""
    have = [s for s in ss if s['row'].get('sparkdash')]
    if not have:
        return y
    cells = {}
    for s in have:
        for key, v in s['row']['sparkdash'].items():
            m = re.fullmatch(r'(.+)x(\d+)', key)
            cells.setdefault(m.group(1), {}).setdefault(s['id'], {})[int(m.group(2))] = v
    names = [c for c in WORKLOADS if c in cells] + sorted(c for c in cells if c not in WORKLOADS)
    panel_title(svg, x, y, 'sparkDash: mean decode rate per stream (tok/s), by concurrent streams',
                "a Python reproduction of MiaAI-Lab/sparkDash's decode-benchmark protocol")
    prov = []
    for s in have:
        r = s['row']
        rounds = f"median of {r['sparkdash_rounds']} rounds" if r.get('sparkdash_rounds') else None
        win = r.get('sparkdash_window')
        where = 'same window as RigMark' if win == r['lab_receipt'] else (f'window {win}' if win else None)
        prov.append(s['tp'] + ': ' + ', '.join(p for p in (rounds, where) if p))
    svg.text(x, y + 34, '; '.join(prov), 11.5, fill='ink2')
    top, step = scale_max(max(v for c in cells.values() for d in c.values() for v in d.values()))
    gap = 18
    cw = (w - 34 - gap * (len(names) - 1)) / len(names)
    height, base = 140, y + 210
    for i, name in enumerate(names):
        cx0 = x + 34 + i * (cw + gap)
        svg.text(cx0 + 12, y + 58, name, 12.5, weight='600')
        vals = {s['id']: cells[name].get(s['id'], {}) for s in have}
        xs = sorted({k for v in vals.values() for k in v})
        line_chart(svg, [s for s in have if vals[s['id']]], vals, cx0, cx0 + cw, base, height, top, step, xs,
                   lambda v: f'{v:.0f}', lambda s, k, v, name=name: f"{s['tp']} {name}, {k} streams: {v:.2f} tok/s",
                   ylabels=(i == 0), unit='tok/s' if i == 0 else None)
    svg.text(x + 34, base + 37, 'concurrent streams', 11, fill='ink2')
    return base + 40


def render_svg(theme, root=ROOT):
    ss, run = series(root)
    svg = Svg(theme)
    t = svg.t
    pad = 24
    svg.text(pad, 36, 'RiNGSiDE: GLM-5.3-Flash NVFP4 on DGX Sparks, measured results', 17, weight='700')
    y = 62
    for s in ss:
        svg.swatch(pad, y - 10, s['slot'])
        svg.text(pad + 19, y, s['label'], 12.5, weight='600')
        for line in s['detail']:
            y += 16
            svg.text(pad + 19, y, line, 11.5, fill='ink2')
        y += 24
    y += 8
    left = WIDTH - 2 * pad
    wa = 548
    end_a = decode_panel(svg, ss, pad, y, wa)
    tool_panel(svg, ss, run, pad + wa + 28, y, left - wa - 28)
    y = end_a + 26
    wb = (left - 32) / 2
    end_b = prefill_panel(svg, ss, pad, y, wb)
    aggregate_panel(svg, ss, pad + wb + 32, y, wb)
    y = end_b + 26
    y = sparkdash_panel(svg, ss, pad, y, left) + 14
    svg.text(pad, y, 'Measured values from bench/results/results.json and bench/results/evidence; rendered by '
                     'bench/results/chart.py. Full tables and caveats: bench/results/README.md.', 11, fill='ink2')
    height = int(y + 18)
    title = 'RiNGSiDE measured results'
    desc = alt_text(root)
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
            f'viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="chart-title chart-desc" '
            f'font-family="{FONT}">\n'
            f'<title id="chart-title">{escape(title)}</title>\n<desc id="chart-desc">{escape(desc)}</desc>\n'
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="10" fill="{t["surface"]}" '
            f'stroke="{t["border"]}"/>\n')
    return head + '\n'.join(svg.parts) + '\n</svg>\n'


# ------------------------------------------------------------------------------------------------------ README block

def alt_text(root=ROOT):
    ss, run = series(root)
    out = []
    for s in ss:
        r = s['row']
        d = r['decode']
        suite = r.get('prefill_suite')
        cold = {int(k): v['cold_s'] for k, v in suite.items()} if suite else {int(k): v for k, v in r['cold'].items()}
        agg = {int(k): v for k, v in r['agg'].items()}
        lo, hi = min(cold), max(cold)
        a0, a1 = min(agg), max(agg)
        part = (f"{s['label']}: single-stream decode {d['code']:.1f} code, {d['prose']:.1f} prose, "
                f"{d['structured']:.1f} structured tok/s; cold prefill {cold[lo]:.1f} s at {depth_label(lo)} to "
                f"{cold[hi]:.1f} s at {depth_label(hi)}; short code aggregate {agg[a0]:.1f} tok/s at {a0} stream to "
                f"{agg[a1]:.1f} tok/s at {a1} streams")
        sd = r.get('sparkdash') or {}
        if 'codex1' in sd:
            part += f"; sparkDash code, 1 stream {sd['codex1']:.1f} tok/s"
        if s is ss[0] and run:
            part += '; tool-eval-bench ' + tool_score_text(run)
        out.append(part)
    return 'RiNGSiDE measured results. ' + '. '.join(out) + '.'


def block(root=ROOT):
    alt = escape(alt_text(root), {'"': '&quot;'})
    return '\n'.join([
        START,
        '<picture>',
        '  <source media="(prefers-color-scheme: dark)" srcset="bench/results/chart-dark.svg">',
        '  <source media="(prefers-color-scheme: light)" srcset="bench/results/chart-light.svg">',
        f'  <img alt="{alt}" src="bench/results/chart-light.svg" width="{WIDTH}">',
        '</picture>',
        '',
        "RiNGSiDE's own measured results only, rendered by `bench/results/chart.py` from `bench/results/results.json`",
        'and the records in `bench/results/evidence/`; the full tables, the caveats and the comparison recipes are in',
        '[bench/results](bench/results/README.md).',
        END,
    ])


def readme_with_block(text, root=ROOT):
    new = block(root)
    if START in text and END in text:
        a = text.index(START)
        b = text.index(END, a) + len(END)
        return text[:a] + new + text[b:]
    m = re.search(r'^## Status\b.*$', text, re.M)
    table = re.search(r'^\|.*$(?:\n\|.*$)*', text[m.end():], re.M) if m else None
    if table is None:
        raise SystemExit('README.md: no table under a "## Status" heading to place the chart after')
    at = m.end() + table.end()
    return text[:at] + '\n\n' + new + text[at:]


def expected(root=ROOT):
    """{path: text} of every generated file as the data gives it."""
    out = {HERE.relative_to(ROOT).as_posix() + '/' + name: render_svg(theme, root) for theme, name in OUTPUTS.items()}
    out['README.md'] = readme_with_block((root / 'README.md').read_text(), root)
    return out


def check(root=ROOT):
    """The generated files that differ from what the data gives (empty when all match)."""
    stale = []
    for rel, text in expected(root).items():
        path = root / rel
        if not path.exists() or path.read_text() != text:
            stale.append(rel)
    return stale


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true', help='write the SVGs and the README block')
    mode.add_argument('--check', action='store_true', help='exit 1 if a committed file differs from the data')
    args = ap.parse_args(argv)
    if args.check:
        stale = check()
        for rel in stale:
            print(f'{rel}: differs from what the data gives; run python3 bench/results/chart.py --write')
        return 1 if stale else 0
    for rel, text in expected().items():
        path = ROOT / rel
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            print('wrote', rel)
    return 0


if __name__ == '__main__':
    sys.exit(main())
