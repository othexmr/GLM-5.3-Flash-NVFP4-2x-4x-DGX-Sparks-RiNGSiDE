# SPDX-License-Identifier: Apache-2.0
"""TP4 fabric preflight: run on each node before the first launch (read-only; it only reads sysfs).

    python3 fabric/preflight_tp4.py --rank N [--topology fabric/topology.example.json] [--gid-index 3]

The TP4 transports hard-code their devices: the lean all-reduce (src/tp4/glm53_lean_allreduce/lean_tp4.py,
HCA_NAMES) and the prefill RDMA ring (src/tp4/glm53_prefill_rdma/protocol.py, HCA_NAMES) open the four PCIe functions
rocep1s0f0, rocep1s0f1, roceP2p1s0f0 and roceP2p1s0f1, and both use GID index NCCL_IB_GID_INDEX (default 3, which
the TP4 profile does not set) on every one of them. The NCCL build accepts only IPv4 GIDs in the four cable subnets.
This script checks, for this node's rank in the ring (fabric/README.md, topology.example.json):

- each of the four devices exists and its port 1 is ACTIVE on an Ethernet link layer;
- the GID at the index the transports use is a RoCE v2 entry holding an IPv4-mapped address;
- that address lies in the subnet of the cable the function is attached to.

Exit 0 when every check passes, 1 otherwise; the report is printed as JSON.
"""
import argparse
import ipaddress
import json
from pathlib import Path
import sys

HCA_NAMES = ('rocep1s0f0', 'rocep1s0f1', 'roceP2p1s0f0', 'roceP2p1s0f1')
HERE = Path(__file__).resolve().parent


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def expected_subnets(topology, rank):
    """{device: subnet} for this rank: both lanes of each cable's port carry an address in that cable's subnet."""
    out = {}
    for link in topology['links']:
        if rank in link['ranks']:
            for lane in link['lanes']:
                out[lane] = ipaddress.ip_network(link['subnet'])
    return out


def gid_ipv4(gid):
    """The IPv4 address of an IPv4-mapped GID (::ffff:a.b.c.d), or None."""
    try:
        addr = ipaddress.IPv6Address(gid)
    except ValueError:
        return None
    return addr.ipv4_mapped


def check(sysfs, topology, rank, gid_index):
    subnets = expected_subnets(topology, rank)
    report = {'rank': rank, 'gid_index': gid_index, 'devices': {}, 'problems': []}
    for name in HCA_NAMES:
        port = Path(sysfs) / name / 'ports' / '1'
        entry = {'present': (Path(sysfs) / name).is_dir()}
        report['devices'][name] = entry
        if not entry['present']:
            report['problems'].append(f'{name}: RDMA device missing')
            continue
        entry['state'] = read(port / 'state')
        entry['link_layer'] = read(port / 'link_layer')
        entry['gid'] = read(port / 'gids' / str(gid_index))
        entry['gid_type'] = read(port / 'gid_attrs' / 'types' / str(gid_index))
        entry['netdev'] = read(port / 'gid_attrs' / 'ndevs' / str(gid_index))
        if not (entry['state'] or '').endswith('ACTIVE'):
            report['problems'].append(f'{name}: port 1 is not ACTIVE ({entry["state"]})')
        if entry['link_layer'] != 'Ethernet':
            report['problems'].append(f'{name}: link layer {entry["link_layer"]}, RoCE needs Ethernet')
        if entry['gid_type'] != 'RoCE v2':
            report['problems'].append(f'{name}: GID {gid_index} is {entry["gid_type"]!r}, the transports need RoCE v2')
        ipv4 = gid_ipv4(entry['gid'] or '')
        entry['ipv4'] = str(ipv4) if ipv4 else None
        if ipv4 is None:
            report['problems'].append(f'{name}: GID {gid_index} ({entry["gid"]}) is not an IPv4-mapped address')
        elif name not in subnets:
            report['problems'].append(f'{name}: no cable of rank {rank} uses this function (topology)')
        elif ipv4 not in subnets[name]:
            report['problems'].append(f'{name}: GID {gid_index} address {ipv4} is outside {subnets[name]}, the subnet '
                                      f'of its cable')
    report['ok'] = not report['problems']
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rank', type=int, required=True, choices=range(4))
    ap.add_argument('--topology', type=Path, default=HERE / 'topology.example.json')
    ap.add_argument('--gid-index', type=int, default=3, help='the NCCL_IB_GID_INDEX the launch uses (default 3)')
    ap.add_argument('--sysfs', type=Path, default=Path('/sys/class/infiniband'))
    a = ap.parse_args()
    report = check(a.sysfs, json.loads(a.topology.read_text()), a.rank, a.gid_index)
    print(json.dumps(report, indent=1))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
