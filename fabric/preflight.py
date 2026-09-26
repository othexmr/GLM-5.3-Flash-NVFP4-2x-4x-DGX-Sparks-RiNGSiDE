#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate the global ring and this node's sysfs. Peer transport is a separate gate."""
import argparse
import json
from pathlib import Path
import preflight_tp4
from ring import validate, digest, cable_subnets


def check(sysfs, topology, rank, gid_index):
    neighbors = validate(topology)
    if rank not in neighbors:
        raise ValueError('rank outside topology')
    report = preflight_tp4.check(sysfs, topology, rank, gid_index)
    domains = {}
    for hca in topology['functions']:
        device = Path(sysfs) / hca / 'device'
        if not device.exists():
            report['problems'].append(f'{hca}: PCI device identity unavailable')
        else:
            domains[hca] = device.resolve().name.split(':')[0]
    if len(domains) == 4:
        if (domains['rocep1s0f0'] != domains['rocep1s0f1']
                or domains['roceP2p1s0f0'] != domains['roceP2p1s0f1']
                or domains['rocep1s0f0'] == domains['roceP2p1s0f0']):
            report['problems'].append('HCA functions do not identify two distinct PCI roots')
    report['pci_domains'] = domains
    for link in topology['links']:
        for hca, expected in link.get('addresses', {}).get(str(rank), {}).items():
            if report['devices'].get(hca, {}).get('ipv4') != expected:
                report['problems'].append(f'{hca}: GID differs from declared endpoint')
    report.update(topology_sha256=digest(topology), ranks=len(neighbors),
                  neighbors=sorted(neighbors[rank]), cable_subnets=cable_subnets(topology),
                  transport_verified=False)
    report['ok'] = not report['problems']
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--topology', type=Path, required=True)
    p.add_argument('--gid-index', type=int, default=3)
    p.add_argument('--sysfs', type=Path, default=Path('/sys/class/infiniband'))
    a = p.parse_args()
    result = check(a.sysfs, json.loads(a.topology.read_text()), a.rank, a.gid_index)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ok'] else 1)
