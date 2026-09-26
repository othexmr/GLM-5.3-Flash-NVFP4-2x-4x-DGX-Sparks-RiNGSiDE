# SPDX-License-Identifier: Apache-2.0
"""fabric/preflight_tp4.py on synthetic sysfs trees: device names, RoCE v2 type and subnet of the GID the TP4
transports use."""
import importlib.util
import ipaddress
import json
from pathlib import Path
import re
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('preflight', ROOT / 'fabric/preflight_tp4.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)
TOPOLOGY = json.loads((ROOT / 'fabric/topology.example.json').read_text())


def mapped(network, host):
    """GID text of the IPv4-mapped address `host` inside `network` (no dotted host address in this file)."""
    v4 = ipaddress.ip_network(network)[host]
    return str(ipaddress.IPv6Address('::ffff:' + str(v4)).exploded)


class Preflight(unittest.TestCase):
    def node(self, rank, gid_index=3, **override):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        subnets = preflight.expected_subnets(TOPOLOGY, rank)
        for name in preflight.HCA_NAMES:
            port = Path(tmp.name) / name / 'ports/1'
            for sub in ('gids', 'gid_attrs/types', 'gid_attrs/ndevs'):
                (port / sub).mkdir(parents=True, exist_ok=True)
            (port / 'state').write_text('4: ACTIVE\n')
            (port / 'link_layer').write_text('Ethernet\n')
            values = {'gid': mapped(subnets[name], 10 + rank), 'type': 'RoCE v2'}
            values.update(override.get(name, {}))
            (port / 'gids' / str(gid_index)).write_text(values['gid'] + '\n')
            (port / 'gid_attrs/types' / str(gid_index)).write_text(values['type'] + '\n')
            (port / 'gid_attrs/ndevs' / str(gid_index)).write_text('eth' + name[-1] + '\n')
        return tmp.name

    def test_every_rank_of_the_ring_passes(self):
        for rank in range(4):
            report = preflight.check(self.node(rank), TOPOLOGY, rank, 3)
            self.assertTrue(report['ok'], report['problems'])

    def test_each_function_belongs_to_its_cable(self):
        subnets = preflight.expected_subnets(TOPOLOGY, 0)
        self.assertEqual(sorted(subnets), sorted(preflight.HCA_NAMES))
        self.assertEqual(str(subnets['rocep1s0f1']), '10.100.224.0/24')
        self.assertEqual(str(subnets['rocep1s0f0']), '10.100.227.0/24')

    def test_missing_device_fails(self):
        sysfs = self.node(1)
        for p in sorted((Path(sysfs) / 'roceP2p1s0f0').rglob('*'), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        (Path(sysfs) / 'roceP2p1s0f0').rmdir()
        report = preflight.check(sysfs, TOPOLOGY, 1, 3)
        self.assertIn('roceP2p1s0f0: RDMA device missing', report['problems'])

    def test_roce_v1_at_the_index_fails(self):
        report = preflight.check(self.node(2, rocep1s0f1={'type': 'IB/RoCE v1'}), TOPOLOGY, 2, 3)
        self.assertTrue(any('need RoCE v2' in p for p in report['problems']), report['problems'])

    def test_link_local_gid_at_the_index_fails(self):
        report = preflight.check(self.node(0, roceP2p1s0f1={'gid': 'fe80:0000:0000:0000:0000:00ff:fe00:0001'}),
                                 TOPOLOGY, 0, 3)
        self.assertTrue(any('not an IPv4-mapped address' in p for p in report['problems']), report['problems'])

    def test_address_of_the_other_cable_fails(self):
        wrong = mapped('10.100.225.0/24', 20)   # rank 0's f1 functions are on the 10.100.224.0/24 cable
        report = preflight.check(self.node(0, rocep1s0f1={'gid': wrong}), TOPOLOGY, 0, 3)
        self.assertTrue(any(re.search(r'rocep1s0f1: .* outside 10\.100\.224\.0/24', p) for p in report['problems']),
                        report['problems'])

    def test_other_gid_index_is_checked_when_the_launch_sets_one(self):
        sysfs = self.node(3, gid_index=5)
        self.assertFalse(preflight.check(sysfs, TOPOLOGY, 3, 3)['ok'])
        self.assertTrue(preflight.check(sysfs, TOPOLOGY, 3, 5)['ok'])

    def test_transport_device_names_match_the_served_sources(self):
        for rel in ('src/tp4/glm53_lean_allreduce/lean_tp4.py', 'src/tp4/glm53_prefill_rdma/protocol.py'):
            text = (ROOT / rel).read_text()
            self.assertIn('HCA_NAMES = ' + repr(preflight.HCA_NAMES).replace("'", '"'), text, rel)


if __name__ == '__main__':
    unittest.main()
