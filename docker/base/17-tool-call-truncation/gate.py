# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
"""Compare the packet test before and after the install inside the built image.

The packet's harness passes 0/1,410 on a Mac with its import shims, but in the real image its
'tool name' check extracts the name from a field the installed engine does not populate, so that
check fails identically before and after the install (1,305 cases on the tested image). This gate
requires: every truncation-related check at zero failures after the install, the 'tool name' count
unchanged (a harness artifact, never a regression hidden by the repair), and no new check failing.
"""
import json,sys
from pathlib import Path

def load(path):
    text=Path(path).read_text();return json.loads(text[text.index('{'):])

before=load(sys.argv[1]);after=load(sys.argv[2])
truncation={'streamed arguments parse as JSON','streaming == non-streaming arguments','parser exposes tool_call_truncated'}
fb=before['failures_by_check'];fa=after['failures_by_check']
problems=[]
if before['mode']!='real' or after['mode']!='real':problems.append('harness did not run against the installed engine')
for check in truncation:
    if fa.get(check,0):problems.append(f'{check}: {fa[check]} failures after install')
    if not fb.get(check,0):problems.append(f'{check}: the pre-install run did not exhibit the defect')
for check,count in fa.items():
    if check not in truncation and count!=fb.get(check,0):problems.append(f'{check}: {fb.get(check,0)} -> {count} (unexpected change)')
if set(fa)-set(fb)-truncation:problems.append('new failing checks: '+str(sorted(set(fa)-set(fb)-truncation)))
print(json.dumps({'before':fb,'after':fa,'checks':after['checks'],'problems':problems}))
if problems:raise SystemExit('STAGE17_GATE_FAIL: '+'; '.join(problems))
print('STAGE17_GATE_PASS')
