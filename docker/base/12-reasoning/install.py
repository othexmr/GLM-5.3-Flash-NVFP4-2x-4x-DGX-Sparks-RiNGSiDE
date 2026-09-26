# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Contributors to this repository
"""Validate every pinned preimage and result before installing GLM53 adapters."""
import argparse,hashlib,json,py_compile
from pathlib import Path
D=Path(__file__).parent

def install(root):
    manifest=json.loads((D/'source-manifest.json').read_text());pending=[]
    for row in manifest['files']:
        p=root/row['path'];before=row['before']
        if before is None:
            if p.exists():raise RuntimeError('Unexpected existing parser: '+str(p))
            result=(D/row['path']).read_bytes()
        else:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=before:raise RuntimeError('Preimage mismatch: '+str(p))
            if row.get('transform')=='required_tool_complete_objects':
                from required_multi_patch import transform
                result=transform(p.read_text()).encode()
            elif row.get('transform')=='required_tool_stream_state':
                from required_stream_patch import transform
                result=transform(p.read_text()).encode()
            elif row.get('transform')=='responses_streaming_order':
                from responses_streaming_patch import transform
                result=transform(p.read_text()).encode()
            elif row.get('transform')=='responses_streamed_identity':
                from responses_identity_patch import transform
                result=transform(p.read_text()).encode()
            elif row.get('transform')=='responses_stored_tool_history':
                from responses_history_patch import transform
                result=transform(p.read_text()).encode()
            else:
                sub=p.parent.name;adapter='Reasoning'if sub=='reasoning'else'Tool'
                module='glm53_moe_reasoning_parser'if sub=='reasoning'else'glm53_moe_tool_parser'
                text=p.read_text();anchor='    "glm47": ('
                if text.count(anchor)!=1:raise RuntimeError('Ambiguous registry anchor: '+str(p))
                registration=f'    "glm53": (\n        "{module}",\n        "Glm53MoeParser{adapter}Adapter",\n    ),\n'
                result=text.replace(anchor,registration+anchor).encode()
        if hashlib.sha256(result).hexdigest()!=row['after']:raise RuntimeError('Result mismatch: '+str(p))
        compile(result,str(p),'exec');pending.append((p,result))
    for p,result in pending:
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(result);py_compile.compile(str(p),doraise=True)
    print('Installed',len(pending),'hash-verified parser files')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages'));install(parser.parse_args().site)
